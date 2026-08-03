# user_management.py
import os
import bcrypt
from datetime import datetime, timezone
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

VALID_ROLES = {"guest", "admin"}

client = MongoClient(os.getenv("MONGO_URI_ADMIN"))
users = client[os.getenv("MONGO_DB_NAME", "portfolio_rag")]["users"]

# Create a unique index on username — prevents duplicates.
# Guarded: an unreachable Atlas at import time should not crash app startup.
try:
    users.create_index("username", unique=True)
except Exception as _e:
    print(f"  user_management: could not ensure username index ({_e})")


def create_user(username: str, password: str, role: str = "guest") -> bool:
    """
    Create a new user. Returns True on success, False if the user already
    exists or the role is invalid (callers show the failure to the admin).
    role: "guest" (read only) or "admin" (can upload)
    """
    if role not in VALID_ROLES:
        print(f"  Invalid role '{role}' — must be one of {sorted(VALID_ROLES)}.")
        return False
    if users.find_one({"username": username}):
        print(f"  User '{username}' already exists.")
        return False

    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
    users.insert_one({
        "username": username,
        "password_hash": hashed,
        "role": role,
        "created_at": datetime.now(timezone.utc),
    })
    print(f"  Created {role} user: {username}")
    return True


def delete_user(username: str) -> bool:
    result = users.delete_one({"username": username})
    if result.deleted_count:
        print(f"  Deleted user: {username}")
        return True
    print(f"  User '{username}' not found.")
    return False


def list_users():
    print(f"\n{'Username':<20} {'Role':<10}")
    print("-" * 30)
    for u in users.find({}, {"username": 1, "role": 1}):
        print(f"  {u['username']:<20} {u.get('role', '?'):<10}")


def change_password(username: str, new_password: str) -> bool:
    hashed = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt())
    result = users.update_one(
        {"username": username},
        {"$set": {"password_hash": hashed}}
    )
    # matched_count, not modified_count — "found" is the success criterion
    if result.matched_count:
        print(f"  Password updated for: {username}")
        return True
    print(f"  User '{username}' not found.")
    return False


def change_role(username: str, new_role: str) -> bool:
    # Whitelist: a typo'd role (e.g. "Admin") would silently lock the user out,
    # because auth_helper.is_admin compares against the exact string "admin".
    if new_role not in VALID_ROLES:
        print(f"  Invalid role '{new_role}' — must be one of {sorted(VALID_ROLES)}.")
        return False
    result = users.update_one(
        {"username": username},
        {"$set": {"role": new_role}}
    )
    if result.matched_count:  # matched — already-set roles are not "not found"
        print(f"  Role updated for {username} -> {new_role}")
        return True
    print(f"  User '{username}' not found.")
    return False
