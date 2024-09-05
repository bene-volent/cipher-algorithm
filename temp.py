import pandas as pd


data = [
    [0.1, 0.2, 0.3, 0.4, 'AES'],
    [0.5, 0.6, 0.7, 0.8, 'DES'],
    [0.9, 1.0, 1.1, 1.2, 'RSA'],
]
print(pd.DataFrame(data).to_numpy().tolist()) # This line will raise an error

