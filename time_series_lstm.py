import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense

# Generate sine wave data
def generate_sine_data(seq_length, n_samples):
    X = []
    y = []
    for _ in range(n_samples):
        start = np.random.rand()
        print(start)
        print(np.sin(start))
        x = np.linspace(start, start + 3 * np.pi, seq_length + 1)
        sequence = np.sin(x)
        print(sequence)
        X.append(sequence[:-1])
        y.append(sequence[-1])
    return np.array(X), np.array(y)

# Prepare data
seq_length = 10
n_samples = 1
X, y = generate_sine_data(seq_length, n_samples)

# print(X)