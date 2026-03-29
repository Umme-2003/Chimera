# DEMO FILE: Project Chimera Vulnerability Showcase
# This file intentionally contains 7 security vulnerabilities for demonstration
# Each vulnerability is documented with comments showing the issue

import os
import pickle
import yaml
import hashlib
from sqlalchemy import text

# ═══════════════════════════════════════════════════════════════
# VULNERABILITY 1: HARDCODED SECRETS (Severity: HIGH)
# ═══════════════════════════════════════════════════════════════
# Risk: API keys exposed in source code can be leaked
# If this code is pushed to a public repo, attackers can steal the key

class PaymentGateway:
    def __init__(self):
        # 🔴 VULNERABLE: Hardcoded API key
        self.stripe_api_key = "FAKE_STRIPE_KEY_FOR_DEMO_ONLY"
        self.database_password = "FAKE_DB_PASSWORD_FOR_DEMO"
        self.jwt_secret = "FAKE_JWT_SECRET_FOR_DEMO"
        self.endpoint = "https://api.stripe.com/v1/"

    def process_payment(self, amount, card_number):
        # This would make an API call with the hardcoded key
        print(f"Processing ${amount} payment")
        return {"status": "success", "amount": amount}


# ═══════════════════════════════════════════════════════════════
# VULNERABILITY 2: SQL INJECTION (Severity: CRITICAL)
# ═══════════════════════════════════════════════════════════════
# Risk: Attackers can execute arbitrary SQL commands
# Example exploit: username = "'; DROP TABLE users; --"

class UserRepository:
    def __init__(self, db_session):
        self.db = db_session

    def get_user_by_username(self, username: str):
        """🔴 VULNERABLE: Using f-string for SQL query construction"""
        query = f"SELECT * FROM users WHERE username = '{username}'"
        # An attacker could pass: ' OR '1'='1' --
        # Resulting query: SELECT * FROM users WHERE username = '' OR '1'='1' --'
        # This returns ALL users, bypassing authentication!
        result = self.db.execute(text(query))
        return result.fetchone()

    def search_users(self, search_term: str):
        """🔴 VULNERABLE: String concatenation in SQL"""
        query = "SELECT * FROM users WHERE name LIKE '%" + search_term + "%'"
        return self.db.execute(text(query)).fetchall()

    def update_user_email(self, user_id: int, new_email: str):
        """🔴 VULNERABLE: Multiple injection points"""
        query = f"UPDATE users SET email = '{new_email}' WHERE id = {user_id}"
        self.db.execute(text(query))
        self.db.commit()


# ═══════════════════════════════════════════════════════════════
# VULNERABILITY 3: CROSS-SITE SCRIPTING (XSS) (Severity: HIGH)
# ═══════════════════════════════════════════════════════════════
# Risk: Attackers can inject malicious scripts
# Example exploit: comment = "<script>alert('XSS')</script>"

class CommentService:
    def __init__(self):
        self.comments = []

    def add_comment(self, user_input: str, author: str):
        """🔴 VULNERABLE: No sanitization before storing"""
        comment = {
            "id": len(self.comments) + 1,
            "content": user_input,  # directly stored without sanitization
            "author": author
        }
        self.comments.append(comment)
        return comment

    def render_comment_html(self, comment_id: int) -> str:
        """🔴 VULNERABLE: Unsanitized output in HTML"""
        for comment in self.comments:
            if comment["id"] == comment_id:
                # Dangerous: user content rendered directly in HTML
                return f"<div class='comment'><p>{comment['content']}</p><span>by {comment['author']}</span></div>"
        return ""

    def render_page(self, user_content: str) -> str:
        """🔴 VULNERABLE: Using innerHTML equivalent"""
        # In a real web framework, this would be dangerous:
        # element.innerHTML = user_content
        return f"""
        <html>
        <body>
            <div id="content">{user_content}</div>
        </body>
        </html>
        """


# ═══════════════════════════════════════════════════════════════
# VULNERABILITY 4: PATH TRAVERSAL (Severity: MEDIUM)
# ═══════════════════════════════════════════════════════════════
# Risk: Attackers can access files outside intended directory
# Example exploit: filename = "../../../etc/passwd"

class FileManager:
    def __init__(self, base_directory: str):
        self.base_dir = base_directory

    def read_user_file(self, filename: str):
        """🔴 VULNERABLE: No path validation"""
        # Attacker can pass: ../../../etc/passwd
        # Full path becomes: /app/uploads/../../../etc/passwd
        # Which resolves to: /etc/passwd (system file!)
        file_path = os.path.join(self.base_dir, filename)
        with open(file_path, "r") as f:
            return f.read()

    def save_upload(self, filename: str, content: bytes):
        """🔴 VULNERABLE: No filename validation"""
        # Attacker can overwrite system files
        file_path = os.path.join(self.base_dir, filename)
        with open(file_path, "wb") as f:
            f.write(content)
        return file_path

    def load_template(self, template_name: str):
        """🔴 VULNERABLE: Direct path construction"""
        template_path = self.base_dir + "/templates/" + template_name
        with open(template_path, "r") as f:
            return f.read()


# ═══════════════════════════════════════════════════════════════
# VULNERABILITY 5: INSECURE DESERIALIZATION (Severity: CRITICAL)
# ═══════════════════════════════════════════════════════════════
# Risk: Remote code execution via malicious serialized objects
# Attackers can craft payloads that execute arbitrary code

class DataImporter:
    def __init__(self):
        self.data = None

    def import_pickle_data(self, pickled_data: bytes):
        """🔴 VULNERABLE: Unsafe pickle.loads"""
        # pickle.loads can execute arbitrary code!
        # Attacker can craft a malicious pickle that runs: rm -rf /
        self.data = pickle.loads(pickled_data)
        return self.data

    def load_yaml_config(self, yaml_content: str):
        """🔴 VULNERABLE: yaml.load without SafeLoader"""
        # yaml.load can execute arbitrary Python code!
        # Example malicious YAML: !!python/object/apply:os.system ["id"]
        config = yaml.load(yaml_content, Loader=yaml.Loader)
        return config

    def process_user_data(self, raw_data: str):
        """🔴 VULNERABLE: Using eval on user input"""
        # NEVER use eval on untrusted input!
        # Attacker can pass: __import__('os').system('rm -rf /')
        return eval(raw_data)


# ═══════════════════════════════════════════════════════════════
# VULNERABILITY 6: WEAK CRYPTOGRAPHY (Severity: MEDIUM)
# ═══════════════════════════════════════════════════════════════
# Risk: Passwords can be cracked, data can be decrypted
# MD5 and SHA1 are too fast and vulnerable to rainbow table attacks

class AuthenticationService:
    def __init__(self):
        self.users = {}
        self.encryption_key = "hardcoded-key-12345"  # Also hardcoded!

    def hash_password_md5(self, password: str) -> str:
        """🔴 VULNERABLE: MD5 for password hashing"""
        # MD5 is deprecated for password hashing
        # It's too fast and vulnerable to brute-force attacks
        return hashlib.md5(password.encode()).hexdigest()

    def hash_password_sha1(self, password: str) -> str:
        """🔴 VULNERABLE: SHA1 for password hashing"""
        # SHA1 is also too fast for password hashing
        return hashlib.sha1(password.encode()).hexdigest()

    def create_user(self, username: str, password: str):
        """🔴 VULNERABLE: Weak password storage"""
        # Using MD5 instead of bcrypt/scrypt/Argon2
        password_hash = self.hash_password_md5(password)
        self.users[username] = password_hash

    def verify_password(self, username: str, password: str) -> bool:
        """Verify password (using weak hash)"""
        if username not in self.users:
            return False
        return self.hash_password_md5(password) == self.users[username]


# ═══════════════════════════════════════════════════════════════
# VULNERABILITY 7: MISSING INPUT VALIDATION (Severity: MEDIUM)
# ═══════════════════════════════════════════════════════════════
# Risk: Unexpected input causes crashes, injection attacks, data corruption

class OrderProcessor:
    def __init__(self):
        self.orders = []

    def create_order(self, user_id, quantity: str, price: str):
        """🔴 VULNERABLE: No input validation"""
        # No validation that quantity is positive integer
        # No validation that price is positive number
        # No validation that user_id exists
        # No sanitization of any inputs

        # Dangerous: direct conversion without validation
        qty = int(quantity)  # Could raise ValueError
        prc = float(price)   # Could raise ValueError

        # Dangerous: negative values accepted
        total = qty * prc

        order = {
            "user_id": user_id,  # Could be any value
            "quantity": qty,    # Could be negative
            "price": prc,       # Could be negative
            "total": total      # Could be negative (refund exploit!)
        }
        self.orders.append(order)
        return order

    def process_refund(self, order_id: str, amount: str):
        """🔴 VULNERABLE: No validation leads to logic errors"""
        # order_id could be "ABC" which would fail
        # amount could be "-100" which would be a double-refund!

        refund_amount = float(amount)  # Negative refund = charging customer!

        # Look up order (will fail if order_id is not int)
        for order in self.orders:
            if str(order.get("id")) == order_id:
                order["refunded"] = refund_amount
                return order

        return None


# ═══════════════════════════════════════════════════════════════
# EXAMPLE USAGE (showing the vulnerabilities in action)
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("CHIMERA SECURITY DEMO: Vulnerable Code Showcase")
    print("=" * 60)

    # 1. Hardcoded secrets
    payment = PaymentGateway()
    print(f"\n1. Hardcoded API Key: {payment.stripe_api_key[:20]}...")

    # 2. SQL Injection
    print("\n2. SQL Injection vulnerability:")
    print("   Safe input: 'john_doe'")
    print("   Malicious:  \"' OR '1'='1' --\"")

    # 3. XSS
    xss_service = CommentService()
    malicious_comment = "<script>alert('XSS')</script>"
    xss_service.add_comment(malicious_comment, "attacker")
    print(f"\n3. XSS - Stored malicious script: {malicious_comment}")

    # 4. Path Traversal
    print("\n4. Path Traversal:")
    print("   Safe filename: 'document.txt'")
    print("   Malicious:    '../../../etc/passwd'")

    # 5. Insecure Deserialization
    print("\n5. Insecure Deserialization:")
    print("   pickle.loads on untrusted data = RCE")
    print("   yaml.load without SafeLoader = RCE")

    # 6. Weak Crypto
    auth = AuthenticationService()
    password_hash = auth.hash_password_md5("password123")
    print(f"\n6. Weak MD5 hash: {password_hash}")

    # 7. Missing Validation
    orders = OrderProcessor()
    try:
        # This will work with invalid input!
        order = orders.create_order("user123", "-10", "-100.00")
        print(f"\n7. Missing Validation: Order total = ${order['total']}")
        print("   (Negative total = refund exploit!)")
    except Exception as e:
        print(f"7. Validation error: {e}")

    print("\n" + "=" * 60)
    print("END OF DEMO")
    print("Run Project Chimera to fix these vulnerabilities!")
    print("=" * 60)
