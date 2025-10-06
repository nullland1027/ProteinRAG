import sys
import os
from main import get_protein_service, initialize_service

sys.path.append('.')


def test_milvus_lite():
    """Test basic Milvus Lite functionality"""
    print("🧪 Starting Milvus Lite tests...")

    # Initialize service
    print("1. Initializing service...")
    if not initialize_service():
        print("❌ Service initialization failed")
        return False

    service = get_protein_service()
    print(f"✅ Service initialized, DB path: {service.db_path}")

    # Test database connection
    print("2. Testing database connection...")
    if not service.connect_database():
        print("❌ Database connection failed")
        return False
    print("✅ Database connected successfully")

    # Test collection creation
    print("3. Testing collection creation...")
    if not service.create_collection_if_not_exists():
        print("❌ Collection creation failed")
        return False
    print("✅ Collection creation succeeded")

    # Get statistics
    print("4. Getting database statistics...")
    stats = service.get_collection_stats()
    print(f"✅ Stats: {stats}")

    # Connection status check
    print("5. Checking connection status...")
    status = service.check_database_connection()
    print(f"✅ Connection status: {status}")

    print("🎉 All tests passed! Milvus Lite configured correctly")
    return True

if __name__ == "__main__":
    test_milvus_lite()
