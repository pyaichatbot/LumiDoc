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
    url = f"{BASE_URL}/login"
    payload = {
        "username": email,
        "password": "testpassword"
    }
    # ✅ Use `data=` instead of `json=`
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    response = requests.post(url, data=payload, headers=headers)
    
    print("📌 Login Response:", response.status_code, response.json())

    if response.status_code == 200:
        data = response.json()
        print("✅ Login successful")
        return data.get("access_token"), data.get("user_id")
    else:
        print("❌ Login failed:", response.text)
        return None, None

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
    
    url = f"{BASE_URL}/upload?chat_id={email}"  # ✅ Include chat_id as query param
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

    # Create fresh test files with proper content
    test_files = [
        ("test1.pdf", "application/pdf", b"PDF content"),  # Binary content for PDF
        ("test2.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", b"DOCX content"),  # Binary content for DOCX
        ("test3.txt", "text/plain", "This is a simple text file.")  # Text content
    ]

    for filename, content_type, content in test_files:
        file_path = os.path.join(test_files_dir, filename)
        try:
            if isinstance(content, str):
                with open(file_path, "w") as f:
                    f.write(content)
            else:
                with open(file_path, "wb") as f:
                    f.write(content)
            print(f"✅ Created test file: {filename}")
        except Exception as e:
            print(f"❌ Error creating {filename}: {e}")
            continue

    # Prepare files for upload
    files = []
    for filename, content_type, _ in test_files:
        file_path = os.path.join(test_files_dir, filename)
        try:
            with open(file_path, "rb") as f:
                files.append(("files", (filename, f.read(), content_type)))
            print(f"✅ Prepared file for upload: {filename}")
        except Exception as e:
            print(f"❌ Error preparing {filename} for upload: {e}")

    if not files:
        print("❌ No files prepared for upload")
        return

    try:
        response = requests.post(url, headers=headers, files=files)
        print(f"📌 Upload Response: {response.status_code} {response.json()}")

        if response.status_code == 200:
            print("✅ File upload successful")
        else:
            print("❌ File upload failed")
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

        # Remove directory if empty
        try:
            if not os.listdir(test_files_dir):
                os.rmdir(test_files_dir)
                print("✅ Removed empty test_files directory")
        except Exception as e:
            print(f"❌ Error removing test_files directory: {e}")

def test_refresh_token(token: str):
    """Test refreshing an access token"""
    url = f"{BASE_URL}/refresh"
    data = {
        "refresh_token": token
    }
    headers = {
        "Content-Type": "application/x-www-form-urlencoded"
    }
    response = requests.post(url, data=data, headers=headers)
    assert response.status_code == 200
    print("✅ Token refreshed:", response.json())

def test_logout(token: str):
    """Test user logout"""
    url = f"{BASE_URL}/logout"
    response = requests.post(url, headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    print("✅ Logout successful:", response.json())

def test_health():
    """Test health check endpoint"""
    url = f"{BASE_URL}/health"
    response = requests.get(url)
    assert response.status_code == 200
    print("✅ API Health Check:", response.json())

def test_retrieve_chat_sessions(token: str, user_id: str, email: str):
    """Test fetching chat sessions"""
    url = f"{BASE_URL}/chat_sessions/{user_id}/{email}"
    response = requests.get(url, headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    chat_sessions = response.json()
    print("✅ Chat Sessions Retrieved:", chat_sessions)
    if chat_sessions:
        global CHAT_ID
        CHAT_ID = chat_sessions[0]["chat_id"]
        global USER_ID
        USER_ID = chat_sessions[0]["user_id"]

def test_delete_chat_session(token: str, user_id: str, email: str):
    """Test deleting a chat session"""
    if not user_id or not email:
        print("⚠️ No user_id or email found, skipping deletion.")
        return
    url = f"{BASE_URL}/chat_sessions/{user_id}/{email}"
    response = requests.delete(url, headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    print("✅ Chat session deleted successfully.")

def test_delete_user(token: str):
    """Test user deletion"""
    url = f"{BASE_URL}/delete_user"
    response = requests.delete(url, headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    print("✅ User account deleted successfully.")

# Run Tests
if __name__ == "__main__":
    test_health()
    random_email = test_signup()
    token, user_id = test_login(random_email)
    test_chat_response(token, user_id, email = random_email)#"testuser6771@example.com"
    #test_upload_files(token, user_id, email = random_email)
    #test_refresh_token(token)    
    #test_retrieve_chat_sessions(token, user_id, email = random_email)
    #test_delete_chat_session(token, user_id, email = random_email)
    #test_logout(token)
    #test_delete_user(token)
    test_health()