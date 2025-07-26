"""
VisionFlow AI - Database Connection Management
=============================================

This module handles database connections, session management, and provides
dependency injection for FastAPI endpoints. Think of this as the "librarian"
of our system - it manages how different parts of the application safely
access and modify data in the database.

Why do we need this layer?
- Connection pooling for performance
- Transaction management for data consistency
- Session lifecycle management to prevent memory leaks
- Database migration support
- Easy testing with temporary databases
"""

import os
import logging
from contextlib import contextmanager
from typing import Generator, Optional

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from .config import get_settings, get_database_engine_kwargs
from .models.database_models import Base, create_all_tables


# =============================================================================
# LOGGER SETUP
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# DATABASE ENGINE SETUP
# =============================================================================

class DatabaseManager:
    """
    Manages database connections and sessions for the application.
    
    This class follows the Singleton pattern - we only want one database
    manager throughout our application to avoid connection conflicts.
    """
    
    def __init__(self):
        self._engine: Optional[Engine] = None
        self._session_factory: Optional[sessionmaker] = None
        self.settings = get_settings()
        
    def initialize(self) -> None:
        """
        Initialize the database engine and session factory.
        
        This method sets up the database connection and prepares everything
        needed for our application to interact with the database.
        """
        if self._engine is not None:
            logger.warning("Database already initialized, skipping...")
            return
            
        logger.info(f"Initializing database connection to: {self._get_safe_db_url()}")
        
        # Get database-specific configuration
        engine_kwargs = get_database_engine_kwargs()
        
        # Create the database engine
        self._engine = create_engine(
            self.settings.database_url,
            **engine_kwargs
        )
        
        # Set up SQLite optimizations if using SQLite
        if "sqlite" in self.settings.database_url.lower():
            self._setup_sqlite_optimizations()
        
        # Create session factory
        self._session_factory = sessionmaker(
            autocommit=False,      # We want explicit transaction control
            autoflush=False,       # We'll flush manually when needed
            bind=self._engine
        )
        
        # Create all tables
        self._create_tables()
        
        logger.info("Database initialization completed successfully")
    
    def _get_safe_db_url(self) -> str:
        """Get database URL without exposing password in logs."""
        url = self.settings.database_url
        if "postgresql" in url and "@" in url:
            # Hide password in PostgreSQL URLs for logging
            parts = url.split("://")[1].split("@")
            if len(parts) == 2:
                user_pass = parts[0].split(":")
                if len(user_pass) == 2:
                    return f"postgresql://{user_pass[0]}:***@{parts[1]}"
        return url if "sqlite" in url else "***"
    
    def _setup_sqlite_optimizations(self) -> None:
        """
        Configure SQLite for better performance and reliability.
        
        SQLite needs some special configuration to work well in web applications:
        - WAL mode for better concurrency
        - Foreign key enforcement
        - Optimized cache settings
        """
        @event.listens_for(self._engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            
            # Enable WAL mode for better concurrency
            cursor.execute("PRAGMA journal_mode=WAL")
            
            # Enable foreign key constraints
            cursor.execute("PRAGMA foreign_keys=ON")
            
            # Optimize cache and synchronization
            cursor.execute("PRAGMA cache_size=10000")  # 10MB cache
            cursor.execute("PRAGMA synchronous=NORMAL")  # Good balance of safety/speed
            cursor.execute("PRAGMA temp_store=memory")   # Keep temp data in memory
            cursor.execute("PRAGMA mmap_size=268435456") # 256MB memory mapping
            
            cursor.close()
            
        logger.info("SQLite optimizations configured")
    
    def _create_tables(self) -> None:
        """
        Create all database tables if they don't exist.
        
        This is safe to call multiple times - it won't recreate existing tables.
        In production, you might use Alembic migrations instead.
        """
        try:
            create_all_tables(self._engine)
            logger.info("Database tables created/verified successfully")
        except Exception as e:
            logger.error(f"Failed to create database tables: {e}")
            raise
    
    def get_session(self) -> Session:
        """
        Get a new database session.
        
        Each session represents a "conversation" with the database.
        Always close sessions when you're done to prevent connection leaks.
        """
        if self._session_factory is None:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        
        return self._session_factory()
    
    @contextmanager
    def get_session_context(self) -> Generator[Session, None, None]:
        """
        Get a database session with automatic cleanup.
        
        This is the preferred way to get database sessions because it
        automatically closes the session even if an error occurs.
        
        Usage:
            with db_manager.get_session_context() as session:
                # Use session here
                user = session.query(User).first()
        """
        session = self.get_session()
        try:
            yield session
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    def close(self) -> None:
        """
        Close all database connections.
        
        Call this when shutting down the application to clean up connections.
        """
        if self._engine:
            self._engine.dispose()
            logger.info("Database connections closed")
    
    @property
    def engine(self) -> Engine:
        """Get the database engine."""
        if self._engine is None:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        return self._engine
    
    def health_check(self) -> dict:
        """
        Check database health and return status information.
        
        This is useful for monitoring and debugging. It tries to execute
        a simple query to verify the database is accessible.
        """
        try:
            with self.get_session_context() as session:
                # Try a simple query
                result = session.execute(text("SELECT 1 as health_check")).fetchone()
                
                # Get some basic stats
                from .models.database_models import get_table_counts
                table_counts = get_table_counts(session)
                
                return {
                    "status": "healthy",
                    "database_type": "sqlite" if "sqlite" in self.settings.database_url else "postgresql",
                    "connection_test": result[0] == 1,
                    "table_counts": table_counts
                }
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }


# =============================================================================
# GLOBAL DATABASE MANAGER INSTANCE
# =============================================================================

# Create a single instance that will be used throughout the application
db_manager = DatabaseManager()


# =============================================================================
# FASTAPI DEPENDENCY INJECTION
# =============================================================================

def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency that provides database sessions to endpoints.
    
    This function is used with FastAPI's dependency injection system.
    It automatically provides a database session to any endpoint that needs it,
    and ensures the session is properly closed after the request.
    
    Usage in FastAPI endpoints:
        @app.get("/users")
        def get_users(db: Session = Depends(get_db)):
            return db.query(User).all()
    """
    with db_manager.get_session_context() as session:
        yield session


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def init_database() -> None:
    """
    Initialize the database. Call this during application startup.
    
    This function sets up the database connection and creates all necessary
    tables. It's designed to be safe to call multiple times.
    """
    try:
        db_manager.initialize()
        logger.info("Database initialization completed")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise


def close_database() -> None:
    """
    Close database connections. Call this during application shutdown.
    
    This ensures all database connections are properly closed when the
    application shuts down, preventing connection leaks.
    """
    try:
        db_manager.close()
        logger.info("Database connections closed successfully")
    except Exception as e:
        logger.error(f"Error closing database connections: {e}")


def reset_database() -> None:
    """
    Reset the database by dropping and recreating all tables.
    
    WARNING: This will delete all data! Only use for development/testing.
    """
    if not get_settings().debug:
        raise RuntimeError("Cannot reset database in production mode")
    
    logger.warning("Resetting database - all data will be lost!")
    
    try:
        # Drop all tables
        Base.metadata.drop_all(bind=db_manager.engine)
        logger.info("All tables dropped")
        
        # Recreate all tables
        Base.metadata.create_all(bind=db_manager.engine)
        logger.info("All tables recreated")
        
    except Exception as e:
        logger.error(f"Database reset failed: {e}")
        raise


def create_test_database() -> str:
    """
    Create an in-memory SQLite database for testing.
    
    This returns a database URL that can be used for testing.
    The database exists only in memory and is automatically cleaned up.
    """
    test_db_url = "sqlite:///:memory:"
    
    # Create a temporary engine for the test database
    test_engine = create_engine(
        test_db_url,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool
    )
    
    # Create all tables in the test database
    Base.metadata.create_all(bind=test_engine)
    
    return test_db_url


# =============================================================================
# DATABASE MIGRATION HELPERS
# =============================================================================

def backup_database(backup_path: str) -> None:
    """
    Create a backup of the database (SQLite only).
    
    For PostgreSQL, you would use pg_dump instead.
    """
    settings = get_settings()
    
    if "sqlite" not in settings.database_url.lower():
        raise ValueError("Database backup only supported for SQLite")
    
    import shutil
    
    # Extract the database file path from the URL
    db_file = settings.database_url.replace("sqlite:///", "")
    
    if os.path.exists(db_file):
        shutil.copy2(db_file, backup_path)
        logger.info(f"Database backed up to {backup_path}")
    else:
        logger.error(f"Database file not found: {db_file}")


def get_database_size() -> dict:
    """
    Get information about database size and usage.
    
    This is useful for monitoring storage usage and planning capacity.
    """
    settings = get_settings()
    
    if "sqlite" in settings.database_url.lower():
        db_file = settings.database_url.replace("sqlite:///", "")
        if os.path.exists(db_file):
            size_bytes = os.path.getsize(db_file)
            return {
                "type": "sqlite",
                "file_path": db_file,
                "size_bytes": size_bytes,
                "size_mb": round(size_bytes / (1024 * 1024), 2)
            }
    
    # For PostgreSQL, you would query system tables to get size info
    return {"type": "postgresql", "size_info": "Use pg_size_database() to get size"}