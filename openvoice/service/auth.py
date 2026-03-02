import hashlib
import hmac
import secrets
from dataclasses import dataclass

from .storage import Storage


@dataclass
class AuthenticatedUser:
    user_id: int
    username: str


class AuthService:
    def __init__(self, storage: Storage) -> None:
        self.storage = storage

    @staticmethod
    def _hash_password(password: str, salt: str) -> str:
        digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000)
        return digest.hex()

    def register(self, username: str, password: str) -> AuthenticatedUser:
        if len(username.strip()) < 3 or len(password) < 8:
            raise ValueError("Username must be >=3 chars and password >=8 chars")
        existing = self.storage.get_user_by_username(username)
        if existing:
            raise ValueError("User already exists")
        salt = secrets.token_hex(16)
        pw_hash = f"{salt}${self._hash_password(password, salt)}"
        user_id = self.storage.create_user(username, pw_hash)
        return AuthenticatedUser(user_id=user_id, username=username)

    def login(self, username: str, password: str) -> str:
        user = self.storage.get_user_by_username(username)
        if not user:
            raise ValueError("Invalid username/password")
        salt, expected = user["password_hash"].split("$", 1)
        actual = self._hash_password(password, salt)
        if not hmac.compare_digest(expected, actual):
            raise ValueError("Invalid username/password")
        token = secrets.token_urlsafe(32)
        self.storage.create_session(token, int(user["id"]))
        return token

    def require_user(self, token: str) -> AuthenticatedUser:
        if not token:
            raise ValueError("Authentication required")
        session = self.storage.get_session(token)
        if not session:
            raise ValueError("Invalid or expired session")
        user_id = int(session["user_id"])
        with self.storage.connection() as conn:
            user = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        if not user:
            raise ValueError("User not found")
        return AuthenticatedUser(user_id=user_id, username=user["username"])
