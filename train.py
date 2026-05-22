import os
import urllib.request
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Ensure directories exist
os.makedirs('models', exist_ok=True)
os.makedirs('visualizations', exist_ok=True)
os.makedirs('data', exist_ok=True)

print("Phase 3: Dataset Collection")
# 1. Dataset Collection (Modern AI & Tech Corpus)
print("Loading Modern AI Dataset...")
text = """
Artificial Intelligence is transforming industries and the world.
Deep learning models solve complex problems like the vanishing gradient.
Traditional RNN networks fail to remember long sequences of data.
LSTM solves the vanishing gradient problem using memory cells and gates.
Machine learning is the future of software development and automation.
Natural language processing enables computers to understand human text.
The vanishing gradient problem occurs during backpropagation through time.
Long short term memory networks use a forget gate to discard useless information.
Data science and artificial intelligence are revolutionizing modern technology.
Neural networks are inspired by the human brain and biological neurons.
Artificial intelligence is the ability of a computer program or a machine to think and learn.
Deep learning is a subset of machine learning based on artificial neural networks.
When gradients vanish the neural network stops learning effectively.
Recurrent neural networks process sequential data like text and time series.
"""
# Repeat the corpus a few times to give the model more epochs of exposure
text = text * 50
text = text.lower()
print(f"Corpus length: {len(text)} characters")

print("\nPhase 4: Data Preprocessing")
# Split into sentences or lines
corpus = text.split('\n')
# Remove empty lines
corpus = [line for line in corpus if line.strip() != ""]

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1
print(f"Total vocabulary size: {total_words}")

# Save tokenizer
with open('models/tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Create input sequences
input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# Pad sequences
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# Create predictors and labels
X, y = input_sequences[:,:-1], input_sequences[:,-1]
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"Max sequence length: {max_sequence_len}")

# Shared parameters
embedding_dim = 50
epochs = 50
batch_size = 64

print("\nPhase 5: Build and Train Simple RNN Model")
# RNN Model
rnn_model = Sequential([
    Embedding(total_words, embedding_dim, input_length=max_sequence_len-1),
    SimpleRNN(100, return_sequences=False),
    Dropout(0.2),
    Dense(total_words, activation='softmax')
])

rnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
rnn_model.summary()

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train RNN
print("Training RNN Model...")
rnn_history = rnn_model.fit(X, y, epochs=epochs, batch_size=batch_size, 
                            validation_split=0.2, callbacks=[early_stop], verbose=1)
rnn_model.save('models/rnn_model.h5')


print("\nPhase 6: Build and Train LSTM Model")
# LSTM Model
lstm_model = Sequential([
    Embedding(total_words, embedding_dim, input_length=max_sequence_len-1),
    LSTM(100, return_sequences=False),
    Dropout(0.2),
    Dense(total_words, activation='softmax')
])

lstm_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
lstm_model.summary()

# Train LSTM
print("Training LSTM Model...")
lstm_history = lstm_model.fit(X, y, epochs=epochs, batch_size=batch_size, 
                              validation_split=0.2, callbacks=[early_stop], verbose=1)
lstm_model.save('models/lstm_model.h5')

print("\nPhase 7: Comparative Analysis & Visualization")
# Plot and save comparisons
sns.set_style("darkgrid")

# 1. Accuracy Comparison
plt.figure(figsize=(10, 6))
plt.plot(rnn_history.history['accuracy'], label='RNN Train Acc', linestyle='--')
plt.plot(rnn_history.history['val_accuracy'], label='RNN Val Acc')
plt.plot(lstm_history.history['accuracy'], label='LSTM Train Acc', linestyle='--')
plt.plot(lstm_history.history['val_accuracy'], label='LSTM Val Acc')
plt.title('Model Accuracy Comparison: RNN vs LSTM')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.savefig('visualizations/accuracy_comparison.png')
plt.close()

# 2. Loss Comparison
plt.figure(figsize=(10, 6))
plt.plot(rnn_history.history['loss'], label='RNN Train Loss', linestyle='--')
plt.plot(rnn_history.history['val_loss'], label='RNN Val Loss')
plt.plot(lstm_history.history['loss'], label='LSTM Train Loss', linestyle='--')
plt.plot(lstm_history.history['val_loss'], label='LSTM Val Loss')
plt.title('Model Loss Comparison: RNN vs LSTM')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.savefig('visualizations/loss_comparison.png')
plt.close()

# Save metadata (max_sequence_len) for inference
with open('models/metadata.json', 'w') as f:
    import json
    json.dump({'max_sequence_len': max_sequence_len}, f)

print("Training pipeline complete! Models and visualizations saved.")
