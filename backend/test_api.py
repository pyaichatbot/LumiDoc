import os
import random
import string
import requests

BASE_URL = "http://localhost:8000"  # Update if running on a different host
# Sample queries for testing
SAMPLE_QUERIES = [
    "What is AI?",
    "Explain deep learning in simple terms.",
    "How does gradient descent work?",
    "Can you summarize the history of machine learning?",
    "What are the benefits of using Elasticsearch?",
    "How do transformers work in NLP?",
]
# Generate random text of given length
def random_text(length=100):
    return ''.join(random.choices(string.ascii_letters + ' ', k=length))

# ✅ 1️⃣ Test User Registration
def test_signup() -> str:
    url = f"{BASE_URL}/signup"
    random_email = f"testuser{random.randint(1000, 9999)}@example.com" 
    payload = {
        "full_name": "Test User",
        "email": random_email,
        "password": "testpassword"
    }
    response = requests.post(url, json=payload)

    print("📌 Signup Response:", response.status_code)
    if response.status_code == 201:
        data = response.json()
        print("✅ User created:", data)
        return random_email
    else:
        print("❌ Signup failed:", response.text)
        return None

# ✅ 2️⃣ Test User Login & Get Access Token
def test_login(email: str):
    """Test user login & Get Access Token"""
    url = f"{BASE_URL}/login"
    payload = {
        "username": email,
        "password": "testpassword"
    }
    # ✅ Use `data=` instead of `json=`
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    response = requests.post(url, data=payload, headers=headers)
    
    print("📌 Login Response:", response.status_code)

    if response.status_code == 200:
        data = response.json()
        print("✅ Login successful")
        return data.get("access_token"), data.get("user_id"), data.get("refresh_token")
    else:
        print("❌ Login failed:", response.text)
        return None, None, None

# ✅ 3️⃣ Test Chat Response
def test_chat_response(token: str, user_id: str, email: str):
    if not token or not user_id:
        print("❌ No token or user_id found, login first!")
        return
    
    url = f"{BASE_URL}/chat_response"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    for _ in range(5):  # Run 5 tests
        query = random.choice(SAMPLE_QUERIES) + " " + random_text(random.randint(50, 500))
        payload = {
            "chat_id": email,  # Using email as chat_id for testing
            "user_id": user_id,
            "query": query
        }
        response = requests.post(f"{BASE_URL}/chat_response", json=payload, headers=headers)
        print(f"📌 Chat Query [{len(query)} chars]:", response.status_code, response.json())
        if response.status_code == 200:
            data = response.json()
            print("✅ Chat response received:", data)
        else:
            print("❌ Chat request failed:", response.text)  
    
def test_upload_files(token: str, user_id: str, email: str):
    """
    Test file upload endpoint with multiple file types.
    """
    if not token or not user_id:
        print("❌ No token or user_id found, login first!")
        return
    
    url = f"{BASE_URL}/upload"
    headers = {
        "Authorization": f"Bearer {token}"
    }

    # Create test_files directory if it doesn't exist
    test_files_dir = os.path.join(os.path.dirname(__file__), "test_files")
    if not os.path.exists(test_files_dir):
        os.makedirs(test_files_dir)
        print(f"✅ Created test_files directory at {test_files_dir}")
    else:
        # Clean up existing files
        for file in os.listdir(test_files_dir):
            file_path = os.path.join(test_files_dir, file)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"❌ Error deleting {file_path}: {e}")
        print("✅ Cleaned up existing test files")

    # Create test files with more substantial content
    test_files = [
        ("test1.pdf", "application/pdf", b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n3 0 obj\n<<\n/Type /Page\n/Parent 2 0 R\n/Resources <<\n/Font <<\n/F1 4 0 R\n>>\n>>\n/MediaBox [0 0 612 792]\n/Contents 5 0 R\n>>\nendobj\n4 0 obj\n<<\n/Type /Font\n/Subtype /Type1\n/BaseFont /Helvetica\n>>\nendobj\n5 0 obj\n<<\n/Length 44\n>>\nstream\nBT\n/F1 24 Tf\n72 720 Td\n(Test PDF Content) Tj\nET\nendstream\nendobj\nxref\n0 6\n0000000000 65535 f\n0000000009 00000 n\n0000000058 00000 n\n0000000115 00000 n\n0000000254 00000 n\n0000000332 00000 n\ntrailer\n<<\n/Size 6\n/Root 1 0 R\n>>\nstartxref\n427\n%%EOF\n"),
        ("test2.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", b"PK\x03\x04\x14\x00\x00\x00\x00\x00\x00\x00!\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00Test DOCX Content"),
        ("test3.txt", "text/plain", "This is a test text file with some meaningful content.\nIt includes multiple lines\nand some sample text for testing.")
    ]

    try:
        # Prepare files for upload
        files_to_upload = []
        
        for filename, content_type, content in test_files:
            file_path = os.path.join(test_files_dir, filename)
            # Write content to file
            mode = "wb" if isinstance(content, bytes) else "w"
            with open(file_path, mode) as f:
                f.write(content)
            
            # Create tuple for upload
            files_to_upload.append(
                ("files", (
                    filename,
                    open(file_path, "rb"),
                    content_type
                ))
            )

        # Add chat_id as a query parameter
        params = {"chat_id": email}
        
        try:
            # Make the request with prepared files
            response = requests.post(url, headers=headers, files=files_to_upload, params=params)
            print(f"📌 Upload Response: {response.status_code}")
            print(f"📌 Response content: {response.text}")

            if response.status_code == 200:
                print("✅ File upload successful")
            else:
                print(f"❌ File upload failed with status code {response.status_code}")
        finally:
            # Close all file handles
            for _, file_tuple in files_to_upload:
                try:
                    file_tuple[1].close()
                except Exception as e:
                    print(f"❌ Error closing file: {e}")

    except Exception as e:
        print(f"❌ Error during file upload: {e}")
    finally:
        # Clean up test files
        for filename, _, _ in test_files:
            file_path = os.path.join(test_files_dir, filename)
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"✅ Cleaned up: {filename}")
            except Exception as e:
                print(f"❌ Error cleaning up {filename}: {e}")

        # Remove test directory if empty
        try:
            if os.path.exists(test_files_dir) and not os.listdir(test_files_dir):
                os.rmdir(test_files_dir)
                print("✅ Removed test_files directory")
        except Exception as e:
            print(f"❌ Error removing test directory: {e}")

def test_refresh_token(refresh_token: str):
    """Test refreshing an access token"""
    if not refresh_token:
        print("❌ No refresh token provided, skipping refresh test")
        return None, None
        
    url = f"{BASE_URL}/refresh"
    
    # Prepare form data
    data = {
        "refresh_token": refresh_token
    }
    
    # Set proper headers for form data
    headers = {
        "Content-Type": "application/x-www-form-urlencoded"
    }
    
    try:
        response = requests.post(url, data=data, headers=headers)
        print(f"📌 Refresh Token Response: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Token refreshed successfully")
            return result.get("access_token"), result.get("refresh_token")
        else:
            print(f"❌ Token refresh failed: {response.text}")
            return None, None
    except Exception as e:
        print(f"❌ Error during token refresh: {e}")
        return None, None

def test_logout(token: str):
    """Test user logout"""
    if not token:
        print("❌ No token provided, skipping logout")
        return False
        
    url = f"{BASE_URL}/logout"
    headers = {"Authorization": f"Bearer {token}"}
    
    try:
        response = requests.post(url, headers=headers)
        print(f"📌 Logout Response: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ Logout successful:", response.json())
            return True
        else:
            print(f"❌ Logout failed: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error during logout: {e}")
        return False

def test_health():
    """Test health check endpoint"""
    url = f"{BASE_URL}/health"
    response = requests.get(url)
    assert response.status_code == 200
    print("✅ API Health Check:", response.json())

def test_retrieve_chat_sessions(token: str, user_id: str, email: str):
    """Test fetching chat sessions"""
    if not token or not user_id:
        print("❌ No token or user_id found, skipping chat sessions retrieval")
        return

    try:
        # Convert user_id to integer and ensure it's passed as an integer
        user_id_int = int(user_id)
        url = f"{BASE_URL}/chat_sessions/{user_id_int}/{email}"
        headers = {"Authorization": f"Bearer {token}"}
        
        response = requests.get(url, headers=headers)
        print(f"📌 Chat Sessions Response: {response.status_code}")
        
        if response.status_code == 200:
            chat_sessions = response.json()
            print("✅ Chat Sessions Retrieved:", chat_sessions)
            if chat_sessions:
                global CHAT_ID
                CHAT_ID = chat_sessions[0]["chat_id"]
                global USER_ID
                USER_ID = chat_sessions[0]["user_id"]
            return chat_sessions
        else:
            print(f"❌ Failed to retrieve chat sessions: {response.text}")
            return None
    except ValueError as e:
        print(f"❌ Invalid user_id format: {user_id}. Error: {e}")
        return None
    except Exception as e:
        print(f"❌ Error retrieving chat sessions: {e}")
        return None

def test_delete_chat_session(token: str, user_id: str, email: str):
    """Test deleting a chat session"""
    if not user_id or not email:
        print("⚠️ No user_id or email found, skipping deletion.")
        return
    try:
        # Convert user_id to integer
        user_id_int = int(user_id)
        url = f"{BASE_URL}/chat_sessions/{user_id_int}/{email}"
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.delete(url, headers=headers)
        
        print(f"📌 Delete Chat Session Response: {response.status_code}")
        if response.status_code == 200:
            print("✅ Chat session deleted successfully:", response.json())
        else:
            print(f"❌ Failed to delete chat session: {response.text}")
            
    except ValueError as e:
        print(f"❌ Invalid user_id format: {user_id}. Error: {e}")
    except Exception as e:
        print(f"❌ Error deleting chat session: {e}")

def test_delete_user(token: str):
    """Test user deletion"""
    if not token:
        print("❌ No token provided, skipping user deletion")
        return
        
    url = f"{BASE_URL}/delete_user"
    headers = {"Authorization": f"Bearer {token}"}
    
    try:
        response = requests.delete(url, headers=headers)
        print(f"📌 Delete User Response: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ User account deleted successfully:", response.json())
        else:
            print(f"❌ Failed to delete user: {response.text}")
            
    except Exception as e:
        print(f"❌ Error deleting user: {e}")
        return False
    
    return response.status_code == 200

# Run Tests
if __name__ == "__main__":
    test_health()
    random_email = test_signup()
    if random_email:
        # Get initial tokens from login
        access_token, user_id, refresh_token = test_login(random_email)
        if access_token and refresh_token:
            print(f"✅ Initial tokens received - User ID: {user_id}")
            
            # Create a chat session first
            test_chat_response(access_token, user_id, email=random_email)
            
            # Try to refresh the tokens
            new_access_token, new_refresh_token = test_refresh_token(refresh_token)
            if new_access_token and new_refresh_token:
                print("✅ Successfully refreshed tokens")
                access_token = new_access_token
                refresh_token = new_refresh_token
            
            # Now retrieve chat sessions
            chat_sessions = test_retrieve_chat_sessions(access_token, user_id, email=random_email)
            if chat_sessions:
                print("✅ Successfully retrieved chat sessions")
            
            # Continue with other tests
            test_upload_files(access_token, user_id, email=random_email)
            test_delete_chat_session(access_token, user_id, email=random_email)
            
            # Try logout first, then delete user
            if test_logout(access_token):
                print("✅ Logout successful")
            else:
                print("⚠️ Logout skipped or failed")
                
            # Finally delete the user
            if test_delete_user(access_token):
                print("✅ User deletion successful")
            else:
                print("❌ User deletion failed")
    
    test_health()