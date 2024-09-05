import numpy as np
import pandas as pd
from collections import Counter
import math

# Function to convert hex string to byte array
def hex_to_bytes(hex_string):
    return bytes.fromhex(hex_string.replace(' ', ''))

# Function to calculate byte frequency distribution
def byte_frequency(byte_data):
    byte_counts = Counter(byte_data)
    byte_freqs = np.zeros(256)  # 256 possible byte values (0-255)
    total_bytes = len(byte_data)
    
    for byte, count in byte_counts.items():
        byte_freqs[byte] = count / total_bytes  # Normalize by input size
    return byte_freqs

# Function to compute entropy from byte frequencies
def compute_entropy(freqs):
    return -np.sum([p * math.log2(p) for p in freqs if p > 0])

# Function to generate features from a hex input
def generate_features(hex_string):
    byte_data = hex_to_bytes(hex_string)
    
    byte_data_np = np.frombuffer(byte_data, dtype=np.uint8)
    
    # Extract features
    byte_freqs = byte_frequency(byte_data_np)          # Byte frequency distribution (256 features)
    entropy = compute_entropy(byte_freqs)               # Entropy (1 feature)
    block_size = len(byte_data) * 8                     # Block size in bits (1 feature)

    # Byte value statistics
    byte_mean = np.mean(byte_data_np)                    # Mean of byte values
    byte_std = np.std(byte_data_np)                      # Standard deviation of byte values

    # Combine all features into a single array
    features = np.concatenate([byte_freqs, [entropy, block_size, byte_mean, byte_std]])
    
    return features
