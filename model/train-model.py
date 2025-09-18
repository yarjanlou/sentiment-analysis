
import json, os
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dropout, Dense, BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import legacy

# -------------------------
# hyperparameters
# -------------------------
NUM_WORDS = 20000    
MAX_LEN   = 200
EMBED_DIM = 128 

# -------------------------
# 1) load data
# -------------------------
print("Loading IMDB…")
(train_x, train_y), (test_x, test_y) = imdb.load_data(num_words=NUM_WORDS)


# pad/truncate to fixed length
train_x = pad_sequences(train_x, maxlen=MAX_LEN, padding='post', truncating='post')
test_x  = pad_sequences(test_x,  maxlen=MAX_LEN, padding='post', truncating='post')

# -------------------------
# 2) build simple LSTM model
# -------------------------
input_layer = tf.keras.layers.InputLayer(
    batch_input_shape=(None, MAX_LEN),
    dtype="int32",
    name="input_layer"
)
inputs = input_layer.output
x = Embedding(input_dim=NUM_WORDS, output_dim=EMBED_DIM)(inputs)
x = Bidirectional(LSTM(64, dropout=0.3, recurrent_dropout=0.3))(x) 
x = Dense(64, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(32, activation='relu')(x) 
outputs = Dense(1, activation='sigmoid')(x)

model = Model(inputs=inputs, outputs=outputs)

optimizer = legacy.Adam(learning_rate=0.001)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

model.summary()

# -------------------------
# 3) train
# -------------------------
EPOCHS = 15
BATCH  = 64

lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',  
    factor=0.5,         
    patience=2,          
    min_lr=0.0001   
)


history = model.fit(
    train_x, train_y,
    validation_split=0.2,
    epochs=EPOCHS,
    batch_size=BATCH,
    callbacks=[lr_scheduler],
    verbose=1
)

# quick test eval
loss, acc = model.evaluate(test_x, test_y, verbose=0)
print(f"Test accuracy: {acc:.4f}")

# -------------------------
# 4) save keras model
# -------------------------
keras_out = "sentiment_model.h5"
model.save(keras_out)
print(f"Saved model: {keras_out}")

model = tf.keras.models.load_model("sentiment_model.h5")

# -------------------------
# 5) export TF.js model
# -------------------------
import tensorflowjs as tfjs
tfjs_out = "tfjs_out"
os.makedirs(tfjs_out, exist_ok=True)
tfjs.converters.save_keras_model(model, tfjs_out)
print(f"Saved TF.js model to {tfjs_out}/")


# -------------------------
# 6) save vocabulary & metadata for JS tokenizer
# -------------------------
word_index = imdb.get_word_index()
top_words = {word: idx for word, idx in word_index.items() if idx < NUM_WORDS}
with open("vocab.json", "w") as f:
    json.dump(top_words, f)

metadata = {
    "num_words": NUM_WORDS,
    "max_len": MAX_LEN,
    "index_from": 3,
    "pad_id": 0,
    "start_id": 1,
    "oov_id": 2
}
with open("metadata.json", "w") as f:
    json.dump(metadata, f)

print("Saved vocab.json and metadata.json")

