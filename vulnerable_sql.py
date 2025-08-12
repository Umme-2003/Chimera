# A new file with a clear SQL injection vulnerability

from sqlalchemy.sql import text

class UserDAO:
    def __init__(self, db_session):
        self.db = db_session

    def get_user_by_username(self, username: str):
        """
        Retrieves a user by their username.
        This function is vulnerable to SQL Injection.
        """
        # VULNERABILITY: Using an f-string to build a raw SQL query.
        # An attacker could provide a username like: ' OR 1=1; --
        # This would bypass authentication.
        raw_query = f"SELECT * FROM users WHERE username = '{username}'"
        
        print(f"Executing dangerous query: {raw_query}")
        
        # In a real app, this would execute the query.
        # We are just simulating the logic here.
        result = self.db.execute(text(raw_query))
        return result.fetchone()

    def get_safe_user_by_username(self, username: str):
        """
        This is the safe way to do it, for comparison.
        """
        # SAFE: Using parameterized queries with named parameters.
        # The database driver handles sanitizing the input.
        safe_query = text("SELECT * FROM users WHERE username = :username")
        result = self.db.execute(safe_query, {"username": username})
        return result.fetchone()