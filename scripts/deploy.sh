#!/bin/bash

# VisionFlow AI - Production Deployment Script
# ============================================
#
# This script handles the complete deployment of VisionFlow AI to production.
# Think of this as your "launch sequence" - it orchestrates all the steps needed
# to take your development system and deploy it safely to a production environment.
#
# The script handles:
# - Environment validation and security checks
# - Database migrations and backups
# - Service configuration and optimization
# - Health checks and rollback procedures
# - SSL certificate setup and renewal
#
# Usage:
#   ./scripts/deploy.sh [--environment staging|production] [--no-backup] [--dry-run]

set -e  # Exit immediately on any error
set -u  # Exit on undefined variables

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DEPLOYMENT_LOG="$PROJECT_ROOT/deployment.log"
BACKUP_DIR="$PROJECT_ROOT/backups/$(date +%Y%m%d_%H%M%S)"

# Default values
ENVIRONMENT="production"
SKIP_BACKUP=false
DRY_RUN=false
FORCE_DEPLOY=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Logging functions
log() {
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$DEPLOYMENT_LOG"
}

log_info() {
    log "${BLUE}ℹ INFO${NC} $1"
}

log_success() {
    log "${GREEN}✓ SUCCESS${NC} $1"
}

log_warning() {
    log "${YELLOW}⚠ WARNING${NC} $1"
}

log_error() {
    log "${RED}✗ ERROR${NC} $1"
}

log_step() {
    log "${PURPLE}🚀 STEP${NC} $1"
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            --no-backup)
                SKIP_BACKUP=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --force)
                FORCE_DEPLOY=true
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    if [[ "$ENVIRONMENT" != "staging" && "$ENVIRONMENT" != "production" ]]; then
        log_error "Environment must be 'staging' or 'production'"
        exit 1
    fi
}

show_help() {
    cat << EOF
VisionFlow AI Production Deployment Script

This script deploys VisionFlow AI to a production environment with proper
safety checks, backups, and health monitoring.

USAGE:
    $0 [OPTIONS]

OPTIONS:
    --environment staging|production    Target environment (default: production)
    --no-backup                        Skip database backup (not recommended)
    --dry-run                          Show what would be done without executing
    --force                            Skip confirmation prompts
    -h, --help                         Show this help message

EXAMPLES:
    $0                                 # Deploy to production with all safety checks
    $0 --environment staging           # Deploy to staging environment
    $0 --dry-run                       # Preview deployment steps
    $0 --force --no-backup            # Quick deploy without backup (risky!)

SAFETY FEATURES:
    - Automatic database backup before deployment
    - Health checks and rollback on failure
    - Configuration validation
    - SSL certificate verification
    - Performance optimization
    - Service monitoring setup

EOF
}

# Pre-deployment validation
validate_environment() {
    log_step "Validating deployment environment"
    
    # Check if running as appropriate user
    if [[ "$ENVIRONMENT" == "production" && "$EUID" -eq 0 ]]; then
        log_warning "Running as root. Consider using a dedicated service user."
    fi
    
    # Validate required files exist
    local required_files=(
        "$PROJECT_ROOT/docker-compose.yml"
        "$PROJECT_ROOT/requirements.txt"
        "$PROJECT_ROOT/.env"
    )
    
    for file in "${required_files[@]}"; do
        if [[ ! -f "$file" ]]; then
            log_error "Required file missing: $file"
            exit 1
        fi
    done
    
    # Check Docker is available
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed or not in PATH"
        exit 1
    fi
    
    # Validate Docker daemon is running
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        exit 1
    fi
    
    # Check available disk space (need at least 5GB for safe deployment)
    local available_space=$(df "$PROJECT_ROOT" | awk 'NR==2 {print $4}')
    local required_space=$((5 * 1024 * 1024))  # 5GB in KB
    
    if [[ "$available_space" -lt "$required_space" ]]; then
        log_error "Insufficient disk space. Need at least 5GB free."
        exit 1
    fi
    
    log_success "Environment validation passed"
}

# Validate configuration files
validate_configuration() {
    log_step "Validating configuration"
    
    # Check .env file has required variables
    local required_vars=(
        "SECRET_KEY"
        "DATABASE_URL"
        "OPENAI_API_KEY"
    )
    
    for var in "${required_vars[@]}"; do
        if ! grep -q "^${var}=" "$PROJECT_ROOT/.env"; then
            log_error "Missing required environment variable: $var"
            exit 1
        fi
        
        # Check if variable has a value
        local value=$(grep "^${var}=" "$PROJECT_ROOT/.env" | cut -d'=' -f2-)
        if [[ -z "$value" || "$value" == "your_"* ]]; then
            log_error "Environment variable $var is not configured properly"
            exit 1
        fi
    done
    
    # Validate OpenAI API key format
    local openai_key=$(grep "^OPENAI_API_KEY=" "$PROJECT_ROOT/.env" | cut -d'=' -f2-)
    if [[ ! "$openai_key" =~ ^sk-[a-zA-Z0-9]{48}$ ]]; then
        log_warning "OpenAI API key format looks suspicious. Please verify it's correct."
    fi
    
    # Check for production-specific settings
    if [[ "$ENVIRONMENT" == "production" ]]; then
        if grep -q "DEBUG=true" "$PROJECT_ROOT/.env"; then
            log_error "DEBUG mode is enabled in production environment"
            exit 1
        fi
        
        if grep -q "sqlite" "$PROJECT_ROOT/.env"; then
            log_warning "Using SQLite in production. PostgreSQL is recommended."
        fi
    fi
    
    log_success "Configuration validation passed"
}

# Create backup of current system
create_backup() {
    if [[ "$SKIP_BACKUP" == true ]]; then
        log_warning "Skipping backup as requested"
        return 0
    fi
    
    log_step "Creating system backup"
    
    mkdir -p "$BACKUP_DIR"
    
    # Backup database if using PostgreSQL
    if grep -q "postgresql" "$PROJECT_ROOT/.env"; then
        log_info "Creating database backup..."
        
        local db_url=$(grep "^DATABASE_URL=" "$PROJECT_ROOT/.env" | cut -d'=' -f2-)
        local backup_file="$BACKUP_DIR/database_backup.sql"
        
        # Extract database connection details
        local db_host=$(echo "$db_url" | sed -n 's/.*@\([^:]*\):.*/\1/p')
        local db_port=$(echo "$db_url" | sed -n 's/.*:\([0-9]*\)\/.*/\1/p')
        local db_name=$(echo "$db_url" | sed -n 's/.*\/\([^?]*\).*/\1/p')
        local db_user=$(echo "$db_url" | sed -n 's/.*\/\/\([^:]*\):.*/\1/p')
        
        # Create database backup
        PGPASSWORD=$(echo "$db_url" | sed -n 's/.*:\([^@]*\)@.*/\1/p') \
        pg_dump -h "$db_host" -p "$db_port" -U "$db_user" -d "$db_name" > "$backup_file"
        
        if [[ $? -eq 0 ]]; then
            log_success "Database backup created: $backup_file"
        else
            log_error "Database backup failed"
            exit 1
        fi
    fi
    
    # Backup uploaded data
    if [[ -d "$PROJECT_ROOT/data" ]]; then
        log_info "Backing up data directory..."
        tar -czf "$BACKUP_DIR/data_backup.tar.gz" -C "$PROJECT_ROOT" data/
        log_success "Data backup created"
    fi
    
    # Backup configuration
    log_info "Backing up configuration..."
    cp "$PROJECT_ROOT/.env" "$BACKUP_DIR/env_backup"
    cp "$PROJECT_ROOT/docker-compose.yml" "$BACKUP_DIR/docker-compose_backup.yml"
    
    # Create restore script
    cat > "$BACKUP_DIR/restore.sh" << 'EOF'
#!/bin/bash
# VisionFlow AI Backup Restore Script
# This script can restore the system to the state when this backup was created

BACKUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$BACKUP_DIR")")"

echo "🔄 Restoring VisionFlow AI from backup..."
echo "Backup created: $(basename "$BACKUP_DIR")"
echo "Project root: $PROJECT_ROOT"

# Restore configuration
cp "$BACKUP_DIR/env_backup" "$PROJECT_ROOT/.env"
cp "$BACKUP_DIR/docker-compose_backup.yml" "$PROJECT_ROOT/docker-compose.yml"

# Restore data
if [[ -f "$BACKUP_DIR/data_backup.tar.gz" ]]; then
    echo "Restoring data directory..."
    cd "$PROJECT_ROOT"
    tar -xzf "$BACKUP_DIR/data_backup.tar.gz"
fi

# Restore database
if [[ -f "$BACKUP_DIR/database_backup.sql" ]]; then
    echo "Database backup found. Please restore manually using:"
    echo "psql -h HOST -p PORT -U USER -d DATABASE < $BACKUP_DIR/database_backup.sql"
fi

echo "✅ Restore completed. Please restart services."
EOF
    
    chmod +x "$BACKUP_DIR/restore.sh"
    
    log_success "Backup completed: $BACKUP_DIR"
}

# Build and deploy services
deploy_services() {
    log_step "Deploying services"
    
    cd "$PROJECT_ROOT"
    
    # Stop existing services gracefully
    log_info "Stopping existing services..."
    if [[ "$DRY_RUN" == false ]]; then
        docker-compose down --timeout 30 || true
    else
        echo "[DRY RUN] Would run: docker-compose down --timeout 30"
    fi
    
    # Pull latest images
    log_info "Pulling latest base images..."
    if [[ "$DRY_RUN" == false ]]; then
        docker-compose pull
    else
        echo "[DRY RUN] Would run: docker-compose pull"
    fi
    
    # Build services
    log_info "Building application images..."
    if [[ "$DRY_RUN" == false ]]; then
        docker-compose build --no-cache
    else
        echo "[DRY RUN] Would run: docker-compose build --no-cache"
    fi
    
    # Start services
    log_info "Starting services..."
    if [[ "$DRY_RUN" == false ]]; then
        if [[ "$ENVIRONMENT" == "production" ]]; then
            docker-compose --profile production up -d
        else
            docker-compose up -d
        fi
    else
        echo "[DRY RUN] Would run: docker-compose up -d"
    fi
    
    log_success "Services deployment completed"
}

# Run health checks
health_check() {
    log_step "Running health checks"
    
    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY RUN] Would run health checks"
        return 0
    fi
    
    # Wait for services to start
    log_info "Waiting for services to start..."
    sleep 30
    
    # Check backend health
    local backend_url="http://localhost:8000"
    local max_attempts=12
    local attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        log_info "Health check attempt $attempt/$max_attempts..."
        
        if curl -f -s "$backend_url/health" > /dev/null; then
            log_success "Backend health check passed"
            break
        fi
        
        if [[ $attempt -eq $max_attempts ]]; then
            log_error "Backend health check failed after $max_attempts attempts"
            return 1
        fi
        
        sleep 10
        ((attempt++))
    done
    
    # Check database connectivity
    log_info "Checking database connectivity..."
    if docker-compose exec -T backend python -c "
from backend.database import db_manager
health = db_manager.health_check()
print(f'Database status: {health[\"status\"]}')
if health['status'] != 'healthy':
    exit(1)
" &> /dev/null; then
        log_success "Database connectivity check passed"
    else
        log_error "Database connectivity check failed"
        return 1
    fi
    
    # Check SAM service
    log_info "Checking SAM service..."
    if curl -f -s "http://localhost:8001/health" > /dev/null; then
        log_success "SAM service health check passed"
    else
        log_warning "SAM service health check failed (may be normal if model is loading)"
    fi
    
    # Check frontend (if deployed)
    if [[ "$ENVIRONMENT" == "production" ]]; then
        log_info "Checking frontend..."
        if curl -f -s "http://localhost:3000" > /dev/null; then
            log_success "Frontend health check passed"
        else
            log_warning "Frontend health check failed"
        fi
    fi
    
    return 0
}

# Setup monitoring and logging
setup_monitoring() {
    log_step "Setting up monitoring and logging"
    
    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY RUN] Would setup monitoring"
        return 0
    fi
    
    # Ensure log directories exist
    mkdir -p "$PROJECT_ROOT/data/logs"
    
    # Setup log rotation
    if command -v logrotate &> /dev/null; then
        log_info "Configuring log rotation..."
        
        cat > "/tmp/visionflow-logrotate" << EOF
$PROJECT_ROOT/data/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    copytruncate
}
EOF
        
        sudo mv "/tmp/visionflow-logrotate" "/etc/logrotate.d/visionflow"
        log_success "Log rotation configured"
    fi
    
    # Setup process monitoring with systemd (if available)
    if command -v systemctl &> /dev/null && [[ "$ENVIRONMENT" == "production" ]]; then
        log_info "Creating systemd service..."
        
        cat > "/tmp/visionflow.service" << EOF
[Unit]
Description=VisionFlow AI
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=$PROJECT_ROOT
ExecStart=/usr/local/bin/docker-compose --profile production up -d
ExecStop=/usr/local/bin/docker-compose down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
EOF
        
        sudo mv "/tmp/visionflow.service" "/etc/systemd/system/visionflow.service"
        sudo systemctl daemon-reload
        sudo systemctl enable visionflow.service
        
        log_success "Systemd service created and enabled"
    fi
    
    log_success "Monitoring setup completed"
}

# Setup SSL certificates (for production)
setup_ssl() {
    if [[ "$ENVIRONMENT" != "production" ]]; then
        log_info "Skipping SSL setup for non-production environment"
        return 0
    fi
    
    log_step "Setting up SSL certificates"
    
    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY RUN] Would setup SSL certificates"
        return 0
    fi
    
    # Check if certbot is available
    if ! command -v certbot &> /dev/null; then
        log_warning "Certbot not found. SSL certificates must be configured manually."
        log_info "To install certbot: sudo apt-get install certbot python3-certbot-nginx"
        return 0
    fi
    
    # This is a placeholder - in real deployment, you'd configure your domain
    log_warning "SSL certificate setup requires domain configuration."
    log_info "To setup SSL manually:"
    log_info "1. Configure your domain to point to this server"
    log_info "2. Run: sudo certbot --nginx -d yourdomain.com"
    log_info "3. Update nginx configuration in nginx/nginx.conf"
}

# Performance optimization
optimize_performance() {
    log_step "Applying performance optimizations"
    
    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY RUN] Would apply performance optimizations"
        return 0
    fi
    
    # Optimize Docker settings
    log_info "Optimizing Docker settings..."
    
    # Set resource limits in docker-compose override
    cat > "$PROJECT_ROOT/docker-compose.override.yml" << EOF
version: '3.8'
services:
  backend:
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
        reservations:
          memory: 1G
          cpus: '0.5'
    
  sam-service:
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
        reservations:
          memory: 2G
          cpus: '1.0'
          
  postgres:
    deploy:
      resources:
        limits:
          memory: 1G
          cpus: '0.5'
        reservations:
          memory: 512M
          cpus: '0.25'
EOF
    
    # Optimize PostgreSQL if it's being used
    if grep -q "postgresql" "$PROJECT_ROOT/.env"; then
        log_info "Applying PostgreSQL optimizations..."
        
        # These would be applied via docker-compose environment variables
        cat >> "$PROJECT_ROOT/docker-compose.override.yml" << EOF
    environment:
      - POSTGRES_SHARED_BUFFERS=256MB
      - POSTGRES_EFFECTIVE_CACHE_SIZE=1GB
      - POSTGRES_WORK_MEM=4MB
      - POSTGRES_MAINTENANCE_WORK_MEM=64MB
EOF
    fi
    
    log_success "Performance optimizations applied"
}

# Rollback function in case of deployment failure
rollback() {
    log_error "Deployment failed! Initiating rollback..."
    
    if [[ "$SKIP_BACKUP" == true ]]; then
        log_error "No backup available for rollback!"
        return 1
    fi
    
    log_info "Stopping failed deployment..."
    docker-compose down --timeout 30 || true
    
    log_info "Restoring from backup..."
    if [[ -x "$BACKUP_DIR/restore.sh" ]]; then
        "$BACKUP_DIR/restore.sh"
    fi
    
    log_info "Restarting previous version..."
    docker-compose up -d
    
    log_warning "Rollback completed. Please investigate the deployment failure."
}

# Main deployment function
main() {
    # Initialize logging
    echo "🚀 VisionFlow AI Deployment Started" > "$DEPLOYMENT_LOG"
    log_info "Deployment started at $(date)"
    log_info "Environment: $ENVIRONMENT"
    log_info "Dry run: $DRY_RUN"
    
    # Parse command line arguments
    parse_args "$@"
    
    # Show deployment summary
    echo ""
    echo "🚀 VisionFlow AI Deployment"
    echo "=========================="
    echo "Environment: $ENVIRONMENT"
    echo "Backup: $(if [[ "$SKIP_BACKUP" == true ]]; then echo "Disabled"; else echo "Enabled"; fi)"
    echo "Dry run: $(if [[ "$DRY_RUN" == true ]]; then echo "Yes"; else echo "No"; fi)"
    echo ""
    
    # Confirmation prompt (unless forced)
    if [[ "$FORCE_DEPLOY" == false && "$DRY_RUN" == false ]]; then
        read -p "Continue with deployment? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Deployment cancelled by user"
            exit 0
        fi
    fi
    
    # Set up error handling
    trap rollback ERR
    
    # Run deployment steps
    validate_environment
    validate_configuration
    create_backup
    deploy_services
    
    # Only run these steps if basic deployment succeeded
    if health_check; then
        setup_monitoring
        setup_ssl
        optimize_performance
        
        log_success "🎉 Deployment completed successfully!"
        
        # Show post-deployment information
        echo ""
        echo "📋 Post-Deployment Information"
        echo "=============================="
        echo "Backup location: $BACKUP_DIR"
        echo "Logs location: $PROJECT_ROOT/data/logs"
        echo "Health check: http://localhost:8000/health"
        echo "API documentation: http://localhost:8000/docs"
        
        if [[ "$ENVIRONMENT" == "production" ]]; then
            echo "Frontend: http://localhost:3000"
            echo ""
            echo "🔒 Security Reminders:"
            echo "- Configure firewall rules"
            echo "- Set up SSL certificates"
            echo "- Review access logs regularly"
            echo "- Monitor system resources"
        fi
        
        echo ""
        echo "✅ VisionFlow AI is now running in $ENVIRONMENT mode!"
        
    else
        log_error "Health checks failed!"
        exit 1
    fi
    
    # Remove error trap on successful completion
    trap - ERR
}

# Run main function with all arguments
main "$@"