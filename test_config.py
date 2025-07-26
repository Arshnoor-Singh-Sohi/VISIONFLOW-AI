# test_config.py
from backend.config import get_settings, validate_configuration

def test_configuration():
    settings = get_settings()
    print(f"✓ App Name: {settings.app_name}")
    print(f"✓ Database URL: {settings.database_url}")
    print(f"✓ SAM Service URL: {settings.sam_service_url}")
    print(f"✓ Upload Path: {settings.upload_path}")
    
    # Validate configuration
    errors = validate_configuration()
    if errors:
        print("❌ Configuration errors found:")
        for error in errors:
            print(f"  - {error}")
        return False
    else:
        print("✅ Configuration looks good!")
        return True

if __name__ == "__main__":
    test_configuration()