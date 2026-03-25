"""
CrypTAS - AES-256 Encryption/Decryption for model weights
Uses PyCryptodome for AES-CBC encryption with PKCS7 padding.
"""

import os
import json
import base64
import hashlib
import numpy as np
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes


def generate_key(passphrase: str = None) -> bytes:
    """Generate a 32-byte AES-256 key. If passphrase provided, derives key from it."""
    if passphrase:
        return hashlib.sha256(passphrase.encode()).digest()
    return get_random_bytes(32)


def encrypt_weights(weights: dict, key: bytes) -> dict:
    """
    Encrypt model weights dictionary using AES-256-CBC.
    weights: dict of {layer_name: numpy_array}
    Returns: dict of {layer_name: {'ciphertext': b64, 'iv': b64}}
    """
    encrypted = {}
    for layer_name, w_array in weights.items():
        # Serialize numpy array to bytes
        w_bytes = w_array.tobytes()
        shape = list(w_array.shape)
        dtype = str(w_array.dtype)

        # Encrypt
        iv = get_random_bytes(16)
        cipher = AES.new(key, AES.MODE_CBC, iv)
        ciphertext = cipher.encrypt(pad(w_bytes, AES.block_size))

        encrypted[layer_name] = {
            "ciphertext": base64.b64encode(ciphertext).decode(),
            "iv": base64.b64encode(iv).decode(),
            "shape": shape,
            "dtype": dtype
        }
    return encrypted


def decrypt_weights(encrypted: dict, key: bytes) -> dict:
    """
    Decrypt model weights from encrypted dict.
    Returns: dict of {layer_name: numpy_array}
    """
    weights = {}
    for layer_name, enc_data in encrypted.items():
        ciphertext = base64.b64decode(enc_data["ciphertext"])
        iv = base64.b64decode(enc_data["iv"])
        shape = tuple(enc_data["shape"])
        dtype = enc_data["dtype"]

        cipher = AES.new(key, AES.MODE_CBC, iv)
        w_bytes = unpad(cipher.decrypt(ciphertext), AES.block_size)
        weights[layer_name] = np.frombuffer(w_bytes, dtype=dtype).reshape(shape)
    return weights


def save_encrypted_weights(weights: dict, key: bytes, filepath: str):
    """Encrypt and save weights to a JSON file."""
    encrypted = encrypt_weights(weights, key)
    with open(filepath, "w") as f:
        json.dump(encrypted, f)
    print(f"[✓] Encrypted weights saved to {filepath}")


def load_encrypted_weights(filepath: str, key: bytes) -> dict:
    """Load and decrypt weights from a JSON file."""
    with open(filepath, "r") as f:
        encrypted = json.load(f)
    return decrypt_weights(encrypted, key)


if __name__ == "__main__":
    # Quick test
    key = generate_key("test_passphrase")
    dummy_weights = {
        "layer1": np.random.randn(10, 5).astype(np.float32),
        "layer2": np.random.randn(5).astype(np.float32)
    }
    enc = encrypt_weights(dummy_weights, key)
    dec = decrypt_weights(enc, key)
    for k in dummy_weights:
        assert np.allclose(dummy_weights[k], dec[k]), f"Mismatch in {k}"
    print("[✓] AES encryption/decryption test passed!")
