# Test each import individually to find the problematic one
try:
    from backend.config import get_settings, validate_configuration
    print("✓ Config import successful")
except Exception as e:
    print(f"✗ Config import failed: {e}")

try:
    from backend.database import init_database, close_database, get_db
    print("✓ Database import successful")
except Exception as e:
    print(f"✗ Database import failed: {e}")

try:
    from backend.api.router import api_router
    print("✓ API router import successful")
except Exception as e:
    print(f"✗ API router import failed: {e}")

try:
    from backend.utils.logging import setup_logging
    print("✓ Utils logging import successful")
except Exception as e:
    print(f"✗ Utils logging import failed: {e}")

try:
    from backend.utils.helpers import get_app_info
    print("✓ Utils helpers import successful")
except Exception as e:
    print(f"✗ Utils helpers import failed: {e}")