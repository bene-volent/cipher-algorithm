import pandas as pd
from Crypto.Cipher import AES, DES, ChaCha20, Blowfish
from Crypto.PublicKey import RSA
from Crypto.Random import get_random_bytes
from Crypto.Cipher import PKCS1_OAEP
from Crypto.Util.Padding import pad
import binascii
from tqdm import tqdm  # Import tqdm for the progress bar

# Function to convert bytes to hexadecimal string
def bytes_to_hex(byte_data):
    return binascii.hexlify(byte_data).decode('utf-8').upper()
  
# Function to split plaintext into chunks for RSA encryption
def split_plaintext(plaintext, max_chunk_size):
    return [plaintext[i:i + max_chunk_size] for i in range(0, len(plaintext), max_chunk_size)]

# Function to generate encrypted data for a given algorithm and mode
def generate_data_for_algorithm(algorithm, mode=None, num_samples=2000):
    data = []
    
    if algorithm in ['AES', 'ChaCha20', 'Blowfish']:
        key_sizes = [16, 24, 32]  # AES and ChaCha20
        if algorithm == 'Blowfish':
            key_sizes = list(range(4, 57))  # Blowfish key sizes in bytes
    elif algorithm == 'DES':
        key_sizes = [8]  # DES has a fixed key size of 8 bytes
    elif algorithm == 'RSA':
        key_sizes = [128, 256, 512]  # RSA key sizes in bytes (1024, 2048, 4096 bits)

    block_size = {
        'AES': 16,
        'DES': 8,
        'ChaCha20': 64,  # ChaCha20 typically uses a larger block size
        'Blowfish': 8
    }

    for key_size in key_sizes:
        max_chunk_size = key_size - 42  # Adjust for padding overhead

        # Add progress bar for each key size
        for _ in tqdm(range(num_samples), desc=f'Generating data for {algorithm} (key size: {key_size})', unit='sample'):
            # Generate a random plaintext of variable size
            plaintext_size = get_random_bytes(1)[0] % 256  # Random size up to 256 bytes
            plaintext = get_random_bytes(plaintext_size)
            key = get_random_bytes(key_size)
            
            if algorithm == 'AES':
                if mode == 'ECB':
                    cipher = AES.new(key, AES.MODE_ECB)
                    encrypted = cipher.encrypt(pad(plaintext, AES.block_size))
                elif mode == 'CBC':
                    iv = get_random_bytes(AES.block_size)
                    cipher = AES.new(key, AES.MODE_CBC, iv)
                    encrypted = iv + cipher.encrypt(pad(plaintext, AES.block_size))  # Prepend IV to ciphertext
                elif mode == 'CTR':
                    nonce = get_random_bytes(8)
                    cipher = AES.new(key, AES.MODE_CTR, nonce=nonce)
                    encrypted = nonce + cipher.encrypt(plaintext)  # Prepend nonce to ciphertext
                else:
                    raise ValueError("Unsupported AES mode")
                
            elif algorithm == 'DES':
                if mode == 'ECB':
                    cipher = DES.new(key, DES.MODE_ECB)
                    encrypted = cipher.encrypt(pad(plaintext, DES.block_size))
                elif mode == 'CBC':
                    iv = get_random_bytes(DES.block_size)
                    cipher = DES.new(key, DES.MODE_CBC, iv)
                    encrypted = iv + cipher.encrypt(pad(plaintext, DES.block_size))  # Prepend IV to ciphertext
                else:
                    raise ValueError("Unsupported DES mode")
            
            elif algorithm == 'RSA':
                key = RSA.generate(key_size * 8)  # Convert bytes to bits
                public_key = key.publickey()
                cipher_rsa = PKCS1_OAEP.new(public_key)

                # Encrypt in chunks if plaintext is too long
                plaintext_chunks = split_plaintext(plaintext, max_chunk_size)
                encrypted_chunks = [cipher_rsa.encrypt(chunk) for chunk in plaintext_chunks]
                encrypted = b''.join(encrypted_chunks)
            
            elif algorithm == 'ChaCha20':
                nonce = get_random_bytes(32)
                cipher = ChaCha20.new(key=key, nonce=nonce)
                encrypted = nonce + cipher.encrypt(plaintext)  # Prepend nonce to ciphertext
                
            elif algorithm == 'Blowfish':
                cipher = Blowfish.new(key, Blowfish.MODE_ECB)
                encrypted = cipher.encrypt(pad(plaintext, Blowfish.block_size))
                
            else:
                raise ValueError("Unsupported algorithm")
            
            # Convert encrypted data to hex string and store
            cipher_hex = bytes_to_hex(encrypted)
            data.append([cipher_hex, algorithm])
    
    return data

# Function to create dataset
def create_dataset():
    algorithms = ['AES', 'DES', 'RSA', 'ChaCha20', 'Blowfish']
    modes = {
        'AES': ['ECB', 'CBC', 'CTR'],
        'DES': ['ECB', 'CBC'],
        'RSA': [None],
        'ChaCha20': [None],
        'Blowfish': [None]
    }
    all_data = []

    total_samples = 0
    for algo in algorithms:
        for mode in modes[algo]:
            num_samples = 2000
            total_samples += num_samples * len(modes[algo])  # Calculate total samples for progress bar
            
    # Use tqdm for progress bar
    with tqdm(total=total_samples, desc="Generating dataset", unit='sample') as pbar:
        for algo in algorithms:
            for mode in modes[algo]:
                print(f"Generating data for {algo} with mode {mode if mode else 'None'}")
                data = generate_data_for_algorithm(algo, mode=mode, num_samples=300)
                all_data.extend(data)
                pbar.update(2000)  # Update progress bar after each batch of samples
    
    # Create a DataFrame and save to CSV
    df = pd.DataFrame(all_data, columns=['Cipher', 'Algorithm'])
    df.to_csv('cryptographic_algorithms_dataset.csv', index=False)
    print("Dataset saved to cryptographic_algorithms_dataset.csv")

if __name__ == "__main__":
    create_dataset()
