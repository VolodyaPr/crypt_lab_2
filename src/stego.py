import struct
import hashlib
import numpy as np

def bytes_to_bits(data: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))

def bits_to_bytes(bits) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.uint8)
    n = (len(bits) // 8) * 8
    return np.packbits(bits[:n]).tobytes()

def get_physical_indices(height: int, width: int, n_bits: int) -> np.ndarray:
    pos = np.arange(n_bits)
    return (pos // 3) * 4 + (pos % 3)

def make_permutation(password: str, max_bits: int) -> np.ndarray:
    seed = int.from_bytes(hashlib.sha256(password.encode()).digest()[:8], 'big')
    rng = np.random.default_rng(seed)
    return rng.permutation(max_bits)

def embed_data(rgba_arr: np.ndarray, payload: bytes, mode: str, password: str = "") -> np.ndarray:
    height, width, _ = rgba_arr.shape
    max_bits = height * width * 3  # 3 канала из 4

    full_payload = struct.pack('>I', len(payload)) + payload
    bits = bytes_to_bits(full_payload)
    total_bits = len(bits)

    if total_bits > max_bits:
        raise ValueError(f"Сообщение слишком велико: нужно {total_bits} бит, доступно {max_bits} бит.")

    flat = rgba_arr.reshape(-1).copy()

    if mode in ['PlainLSB', 'PlainCipherLSB']:
        phys_indices = get_physical_indices(height, width, total_bits)
    elif mode == 'RandLSB':
        perm = make_permutation(password, max_bits)
        logical_indices = perm[:total_bits]
        phys_indices = (logical_indices // 3) * 4 + (logical_indices % 3)
    else:
        raise ValueError(f"Неизвестный режим: {mode}")

    flat[phys_indices] = (flat[phys_indices] & 0xFE) | bits
    return flat.reshape(rgba_arr.shape)

def extract_data(rgba_arr: np.ndarray, mode: str, password: str = "") -> bytes:
    height, width, _ = rgba_arr.shape
    flat = rgba_arr.reshape(-1)

    if mode in ['PlainLSB', 'PlainCipherLSB']:
        phys_header = get_physical_indices(height, width, 32)
    else:
        perm = make_permutation(password, height * width * 3)
        logical_header = perm[:32]
        phys_header = (logical_header // 3) * 4 + (logical_header % 3)

    header_bits = flat[phys_header] & 1
    length = struct.unpack('>I', bits_to_bytes(header_bits))[0]

    total_bits = 32 + length * 8

    if mode in ['PlainLSB', 'PlainCipherLSB']:
        phys_data = get_physical_indices(height, width, total_bits)
    else:
        logical_data = perm[:total_bits]
        phys_data = (logical_data // 3) * 4 + (logical_data % 3)

    data_bits = flat[phys_data] & 1
    raw = bits_to_bytes(data_bits)
    return raw[4:4 + length]