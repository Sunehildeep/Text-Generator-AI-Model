'''
Import the data and process it.
'''
import tensorflow as tf
import os
import glob
import pandas as pd
import re

# FOR POEMS
# paths = ['depression', 'love']
# text = ""
# for path in paths:
#     files = glob.glob(os.path.join(path, "*.txt"))
#     for file in files:
#         with open(file, 'r', encoding='utf-8') as f:
#             text += f.read()

# FOR OFFICE
# data = pd.read_csv('Office.csv')
# #Append speaker and line together
# #Make speaker upper
# data['speaker'] = data['speaker'].str.upper()

# data['line'] = data['speaker'] + "\n" + data['line'] + "\n"

# FOR SONGS
data = pd.read_csv('lyrics-data.csv', encoding='utf-8')
data = data.dropna()
data = data[data['language'] == 'en']

#Only use 2% of the data (Laptop memory is limited) (Data is too large)
data = data.sample(frac=0.02)
text = data['Lyric'].str.cat(sep='\n')

#Remove special characters except for new line and punctuation '
text = re.sub(r'[^a-zA-Z0-9\'\n\.\,\!\?]', ' ', text)

#Replace two whitespaces with one
text = re.sub(r'  ', ' ', text)

print(text[:1000])

print(f'Length of text: {len(text)} characters')
vocab = sorted(set(text))
print(f'{len(vocab)} unique characters')

ids_from_chars = tf.keras.layers.experimental.preprocessing.StringLookup(
    vocabulary=list(vocab), mask_token=None)

chars_from_ids = tf.keras.layers.experimental.preprocessing.StringLookup(
    vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None)

def text_from_ids(ids):
  return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)

all_ids = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))
ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)
for ids in ids_dataset.take(10):
    print(chars_from_ids(ids).numpy().decode('utf-8'))

seq_length = 100

sequences = ids_dataset.batch(seq_length+1, drop_remainder=True)

for seq in sequences.take(1):
  print(chars_from_ids(seq))

for seq in sequences.take(5):
  print(text_from_ids(seq).numpy())

def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)

'''
Create constants and model
'''
BATCH_SIZE = 512
BUFFER_SIZE = 10000
LR = 1e-3
EPOCHS = 500
TESTING = True
vocab_size = len(ids_from_chars.get_vocabulary())
embedding_dim = 512
n_units = 1024
dropout = 0.1

dataset = (
    dataset
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(tf.data.experimental.AUTOTUNE))

class Model(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, n_units):
    super().__init__(self)
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.lstm_layer = tf.keras.layers.LSTM(n_units, return_sequences=True, return_state=True, dropout=dropout)
    self.dense = tf.keras.layers.Dense(vocab_size)

  def call(self, inputs, states=None, return_state=False, training=False):
    x = inputs
    x = self.embedding(x, training=training)
    x, states, carry_state = self.lstm_layer(x, initial_state=states, training=training)
    x = self.dense(x, training=training)

    if return_state:
      return x, [states, carry_state]
    else:
      return x

class Generator(tf.keras.Model):
  def __init__(self, model, chars_from_ids, ids_from_chars, temperature=1.0):
    super().__init__()
    self.temperature = temperature
    self.model = model
    self.chars_from_ids = chars_from_ids
    self.ids_from_chars = ids_from_chars

    # Create a mask to prevent "[UNK]" from being generated.
    skip_ids = self.ids_from_chars(['[UNK]'])[:, None]
    sparse_mask = tf.SparseTensor(
        # Put a -inf at each bad index.
        values=[-float('inf')]*len(skip_ids),
        indices=skip_ids,
        # Match the shape to the vocabulary
        dense_shape=[len(ids_from_chars.get_vocabulary())])
    self.prediction_mask = tf.sparse.to_dense(sparse_mask)

  @tf.function
  def generate_one_step(self, inputs, states=None):
    # Convert strings to token IDs.
    input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
    input_ids = self.ids_from_chars(input_chars).to_tensor()

    # Run the model.
    # predicted_logits.shape is [batch, char, next_char_logits]
    predicted_logits, states = self.model(inputs=input_ids, states=states,
                                          return_state=True)
    # Only use the last prediction.
    predicted_logits = predicted_logits[:, -1, :]
    predicted_logits = predicted_logits/self.temperature
    # Apply the prediction mask: prevent "[UNK]" from being generated.
    predicted_logits = predicted_logits + self.prediction_mask

    # Sample the output logits to generate token IDs.
    predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
    predicted_ids = tf.squeeze(predicted_ids, axis=-1)

    # Convert from token ids to characters
    predicted_chars = self.chars_from_ids(predicted_ids)

    # Return the characters and model state.
    return predicted_chars, states 

class CustomTraining(Model):
  @tf.function
  def train_step(self, inputs):
      inputs, labels = inputs
      with tf.GradientTape() as tape:
          predictions = self(inputs, training=True)
          loss = self.loss(labels, predictions)
      grads = tape.gradient(loss, model.trainable_variables)
      self.optimizer.apply_gradients(zip(grads, model.trainable_variables))

      return {'loss': loss}   
  
  @tf.function
  def test_step(self, inputs):
      inputs, labels = inputs
      predictions = self(inputs, training=False)
      val_loss = self.loss(labels, predictions)

      return {'val_loss': val_loss}
        
  
model = CustomTraining(
    vocab_size=len(ids_from_chars.get_vocabulary()),
    embedding_dim=embedding_dim,
    n_units=n_units)
     
loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.optimizers.Adam(learning_rate=LR)
model.compile(optimizer = optimizer,
              loss=loss, metrics=['accuracy'])

checkpoint_dir = './training_checkpoints'
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_dir,
    save_weights_only=True)

#Loading the data
model.load_weights(checkpoint_dir + '/ckpt_79')

one_step_model = Generator(model, chars_from_ids, ids_from_chars)

def generate_text(model, start_string):
    num_generate = 1000
    states = None

    # Converting our start string to numbers (vectorizing)
    input_eval = tf.constant([start_string])

    # Empty string to store our results
    text_generated = [input_eval]

    for _ in range(num_generate):
        input_eval, states = model.generate_one_step(input_eval, states=states)

        text_generated.append(input_eval)
    
    text_generated = tf.strings.join(text_generated)

    return text_generated[0].numpy().decode('utf-8')


if TESTING == True:
    while True:
        print("_"*80)
        text = input("Enter a string to start with: ")
        print("_"*80)
        print(generate_text(one_step_model, start_string=text))

else:
    #Split dataset into train and test
    DATASET_SIZE = tf.data.experimental.cardinality(dataset).numpy()
    train_size = int(0.7 * DATASET_SIZE)
    test_size = int(0.15 * DATASET_SIZE)

    train_dataset = dataset.take(train_size)
    test_dataset = dataset.skip(train_size)
    test_dataset = test_dataset.take(test_size)

    #Print test and train dataset size
    print(f"Full dataset size: {tf.data.experimental.cardinality(dataset).numpy()}")
    print(f"Train dataset size: {tf.data.experimental.cardinality(train_dataset).numpy()}")
    print(f"Test dataset size: {tf.data.experimental.cardinality(test_dataset).numpy()}")

    mean = tf.metrics.Mean()
    val_mean = tf.metrics.Mean()
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
    for epoch in range(EPOCHS):

        mean.reset_states()
        val_mean.reset_states()
        for (batch_n, (inp, target)) in enumerate(train_dataset):
            logs = model.train_step([inp, target])
            mean.update_state(logs['loss'])

        for (batch_n, (inp, target)) in enumerate(test_dataset):
            logs = model.test_step([inp, target])
            val_mean.update_state(logs['val_loss'])

        # saving (checkpoint) the model every 5 epochs
        if (epoch + 1) % 5 == 0:
            model.save_weights(checkpoint_prefix.format(epoch=epoch))

        print(f"Epoch {epoch+1} | Train Loss: {mean.result().numpy():.4f} | Val Loss: {val_mean.result().numpy():.4f}")

        print("_"*80)
        print(generate_text(one_step_model, start_string=u"Love "))
        print("_"*80)

    model.save_weights(checkpoint_prefix.format(epoch=epoch))