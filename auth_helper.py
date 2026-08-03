# auth_helper.py
import os
import bcrypt
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

# One client for the process — a new MongoClient per login attempt leaks a
# connection pool (and its monitor threads) each time the form is submitted.
_client = None

# Constant-time-ish dummy check so unknown usernames take as long as known
# ones — otherwise response timing enumerates valid accounts.
_DUMMY_HASH = bcrypt.hashpw(b"dummy-password-for-timing", bcrypt.gensalt())


def get_users_collection():
    global _client
    if _client is None:
        _client = MongoClient(os.getenv("MONGO_URI_ADMIN"))
    return _client[os.getenv("MONGO_DB_NAME", "portfolio_rag")]["users"]


def verify_login(username: str, password: str) -> dict | None:
    """
    Returns the user document if credentials are valid, None otherwise.
    Never returns the password hash.
    """
    users = get_users_collection()
    user = users.find_one({"username": username})

    if not user:
        bcrypt.checkpw(password.encode(), _DUMMY_HASH)  # burn the same time
        return None

    stored_hash = user.get("password_hash")
    if stored_hash is None:
        return None
    if isinstance(stored_hash, str):  # tolerate hashes stored as str via mongosh
        stored_hash = stored_hash.encode()

    if bcrypt.checkpw(password.encode(), stored_hash):
        return {
            "username": user["username"],
            "role":     user.get("role", "guest"),  # missing role → least privilege
        }

    return None


def is_admin(session_state) -> bool:
    return session_state.get("role") == "admin"


def is_authenticated(session_state) -> bool:
    return session_state.get("authenticated", False)
