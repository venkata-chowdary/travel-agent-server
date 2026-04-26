import requests

BASE_URL = "http://127.0.0.1:8000/api/auth"

def test_auth():
    print("Testing signup...")
    signup_data = {"email": "test_auth_user@example.com", "name": "Test User", "password": "password123"}
    res = requests.post(f"{BASE_URL}/signup", json=signup_data)
    print("Signup response:", res.status_code, res.text)

    print("Testing login...")
    login_data = {"email": "test_auth_user@example.com", "password": "password123"}
    res = requests.post(f"{BASE_URL}/login", json=login_data)
    print("Login response:", res.status_code, res.text)
    
    if res.status_code == 200:
        token = res.json().get("access_token")
        print("Testing me...")
        headers = {"Authorization": f"Bearer {token}"}
        res = requests.get(f"{BASE_URL}/me", headers=headers)
        print("Me response:", res.status_code, res.text)

if __name__ == "__main__":
    test_auth()
