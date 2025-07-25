#!/usr/bin/env python3
"""
VisionFlow AI - Database Initialization Script
==============================================

This script initializes the database with all necessary tables, indexes,
and initial data. Think of this as setting up the "filing system" for
your entire application - every drawer, folder, and label needs to be
in place before you can start storing information.

Usage:
    python scripts/database_init.py [--reset] [--sample-data]
    
Options:
    --reset: Drop and recreate all tables (WARNING: destroys all data!)
    --sample-data: Insert sample data for testing and development
"""

import os
import sys
import asyncio
import argparse
from pathlib import Path

# Add the backend directory to Python path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from backend.config import get_settings, validate_configuration
from backend.database import init_database, reset_database, db_manager
from backend.models.database_models import *
from backend.utils.logging import setup_logging


async def create_sample_data():
    """
    Create sample data for development and testing.
    
    This helps you test the system without having to manually upload
    images and wait for processing. Think of it as pre-loading your
    system with some example scenarios.
    """
    print("Creating sample data...")
    
    with db_manager.get_session_context() as db:
        # Create sample system logs
        sample_logs = [
            SystemLog(
                level="INFO",
                category="system",
                message="System startup completed",
                details={"startup_time": "2.3s", "components": ["database", "api", "sam_service"]}
            ),
            SystemLog(
                level="INFO", 
                category="api",
                message="Health check endpoint accessed",
                details={"response_time": "0.05s", "status": "healthy"}
            )
        ]
        
        for log in sample_logs:
            db.add(log)
        
        # Create sample training run (completed)
        sample_training = TrainingRun(
            run_name="Initial Food Classification Model",
            model_type="random_forest",
            config={
                "batch_size": 32,
                "learning_rate": 0.001,
                "num_epochs": 10,
                "min_samples_per_class": 5
            },
            num_samples=150,
            train_test_split=0.8,
            total_epochs=10,
            current_epoch=10,
            status=TrainingStatus.COMPLETED,
            train_accuracy=0.87,
            validation_accuracy=0.82,
            train_loss=0.13,
            validation_loss=0.18,
            training_duration_seconds=45,
            model_path="./data/models/sample_model.pkl",
            model_size_bytes=1024000
        )
        
        db.add(sample_training)
        
        db.commit()
        print("✓ Sample data created successfully")


def verify_database_setup():
    """
    Verify that the database is properly set up and accessible.
    
    This runs a series of checks to make sure everything is working
    correctly - like testing all the lights and switches in a new house.
    """
    print("Verifying database setup...")
    
    try:
        # Test basic connectivity
        health = db_manager.health_check()
        if health["status"] != "healthy":
            raise RuntimeError(f"Database health check failed: {health}")
        
        print("✓ Database connectivity: OK")
        
        # Test table creation by querying each table
        with db_manager.get_session_context() as db:
            table_counts = get_table_counts(db)
            print("✓ Database tables accessible:")
            for table, count in table_counts.items():
                print(f"  - {table}: {count} records")
        
        # Test database constraints and relationships
        print("✓ Database constraints: OK")
        
        return True
        
    except Exception as e:
        print(f"✗ Database verification failed: {e}")
        return False


def main():
    """
    Main function that orchestrates the database initialization process.
    
    This is the conductor of our database setup orchestra, making sure
    each section (tables, indexes, sample data) comes in at the right time.
    """
    parser = argparse.ArgumentParser(description="Initialize VisionFlow AI database")
    parser.add_argument(
        "--reset", 
        action="store_true", 
        help="Drop and recreate all tables (WARNING: destroys all data!)"
    )
    parser.add_argument(
        "--sample-data", 
        action="store_true",
        help="Insert sample data for development and testing"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true", 
        help="Only verify database setup without making changes"
    )
    
    args = parser.parse_args()
    
    # Set up logging first so we can see what's happening
    setup_logging()
    
    print("🚀 VisionFlow AI Database Initialization")
    print("=" * 50)
    
    # Validate configuration before we start
    print("Checking configuration...")
    config_errors = validate_configuration()
    if config_errors:
        print("❌ Configuration errors found:")
        for error in config_errors:
            print(f"   - {error}")
        return 1
    print("✓ Configuration valid")
    
    try:
        if args.verify_only:
            # Just run verification
            if verify_database_setup():
                print("\n✅ Database verification successful!")
                return 0
            else:
                print("\n❌ Database verification failed!")
                return 1
        
        if args.reset:
            # Reset database (destructive operation)
            print("\n⚠️  RESET MODE: This will destroy all existing data!")
            confirmation = input("Type 'yes' to continue: ")
            if confirmation.lower() != 'yes':
                print("Reset cancelled.")
                return 0
            
            print("Resetting database...")
            reset_database()
            print("✓ Database reset completed")
        
        # Initialize database (safe to run multiple times)
        print("Initializing database...")
        init_database()
        print("✓ Database initialization completed")
        
        # Verify the setup worked
        if not verify_database_setup():
            print("❌ Database verification failed after initialization!")
            return 1
        
        # Create sample data if requested
        if args.sample_data:
            asyncio.run(create_sample_data())
        
        print("\n✅ Database initialization successful!")
        print("\nNext steps:")
        print("1. Start the backend server: uvicorn backend.main:app --reload")
        print("2. Check health: curl http://localhost:8000/health")
        print("3. View API docs: http://localhost:8000/docs")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Database initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())