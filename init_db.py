# init_db.py
from backend.database import init_database, reset_database
from backend.config import get_settings

def setup_database():
    try:
        print("🔄 Initializing database...")
        init_database()
        print("✅ Database initialized successfully")
        
        # Verify database connection
        from backend.database import db_manager
        health = db_manager.health_check()
        print(f"📊 Database health: {health['status']}")
        
    except Exception as e:
        print(f"❌ Database setup failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    setup_database()