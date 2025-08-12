import os

class DataService:
    def __init__(self):
        self.api_key = os.environ.get('API_KEY')  # Load API key from environment variable
        self.endpoint = "https://api.dataservice.com/v1/"

    def connect(self):
        print(f"Connecting to {self.endpoint} with key {self.api_key}")
        # In a real app, an HTTP request would be made here
        return True

    def get_user_data(self, user_id):
        if not self.connect():
            return None
        
        print(f"Fetching data for user {user_id}")
        return {"user_id": user_id, "data": "Sample user data"}

if __name__ == "__main__":
    service = DataService()
    user_data = service.get_user_data("user-123")
    print(f"Received data: {user_data}")