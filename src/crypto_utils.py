import os
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from src.config import SALT_LEN, NONCE_LEN, KDF_ITER

def derive_key(password: str, salt: bytes) -> bytes:
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=16, salt=salt, iterations=KDF_ITER)
    return kdf.derive(password.encode())

def encrypt_message(message: bytes, password: str) -> bytes:
    salt = os.urandom(SALT_LEN)
    nonce = os.urandom(NONCE_LEN)
    key = derive_key(password, salt)
    ct = AESGCM(key).encrypt(nonce, message, None)
    return salt + nonce + ct

def decrypt_message(payload: bytes, password: str) -> bytes:
    salt = payload[:SALT_LEN]
    nonce = payload[SALT_LEN:SALT_LEN + NONCE_LEN]
    ct = payload[SALT_LEN + NONCE_LEN:]
    key = derive_key(password, salt)
    return AESGCM(key).decrypt(nonce, ct, None)