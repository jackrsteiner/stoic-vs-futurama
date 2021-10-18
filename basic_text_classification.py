import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import preprocessing

# Check tensorflow version
#print(tf.__version__)

# Set data directory
dataset_dir = './data'
# Check dataset director is properly set
#print(os.listdir(dataset_dir))

batch_size = 32
seed_value = 42
validation_split = 0.2

## Import data from directory

# training dataset
raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
    dataset_dir,
    batch_size=batch_size,
    seed=seed_value,
    validation_split=validation_split,
    subset='training'
)

# validation dataset
raw_val_ds = tf.keras.preprocessing.text_dataset_from_directory(
    dataset_dir,
    batch_size=batch_size,
    seed=seed_value,
    validation_split=validation_split,
    subset='validation',
)

print("Label 0 corresponds to", raw_train_ds.class_names[0])
print("Label 1 corresponds to", raw_train_ds.class_names[1])
print("Label 2 corresponds to", raw_train_ds.class_names[2])

# Number of words to tokenize
max_features = 10000
# Exact number of tokens each sample/example has
sequence_length = 150

vectorize_layer = layers.TextVectorization(
    standardize="lower_and_strip_punctuation",
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length)

# Make a text-only dataset (without labels), then call adapt
train_text = raw_train_ds.map(lambda x, y: x)
vectorize_layer.adapt(train_text)

# Create text vectorization funciton
def vectorize_text(text, label):
  text = tf.expand_dims(text, -1)
  return vectorize_layer(text), label

## view a batch (of 32 reviews and labels) from the dataset
# text_batch, label_batch = next(iter(raw_train_ds))
# first_review, first_label = text_batch[0], label_batch[0]
# print("Review", first_review)
# print("Label", raw_train_ds.class_names[first_label])
# print("Vectorized review", vectorize_text(first_review, first_label))

train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
# test_ds = raw_test_ds.map(vectorize_text)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
# test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

## Create neural network
embedding_dim = 16

# Model Density is 3, the number of labels
model = tf.keras.Sequential([
  layers.Embedding(max_features + 1, embedding_dim),
  layers.Dropout(0.2),
  layers.GlobalAveragePooling1D(),
  layers.Dropout(0.2),
  layers.Dense(3)])

## View model summary
model.summary()

# configure the model to use an optimizer and a loss function
# When compiling the model, change the loss to tf.keras.losses.SparseCategoricalCrossentropy. This is the correct loss function to use for a multi-class classification problem, when the labels for each class are integers (in this case, they can be 0, 1, 2, or 3). In addition, change the metrics to metrics=['accuracy'], since this is a multi-class classification problem (tf.metrics.BinaryAccuracy is only used for binary classifiers).
model.compile(loss=losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer='adam',
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

# Train the model
epochs = 50
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs)

# Evaluate model (should be with test_ds, but I don't have that)
loss, accuracy = model.evaluate(val_ds)

print("Loss: ", loss)
print("Accuracy: ", accuracy)

# # Create plot of accuracy and loss over time
# history_dict = history.history
# history_dict.keys()

# print(history)

# acc = history_dict['accuracy']
# val_acc = history_dict['val_accuracy']
# loss = history_dict['loss']
# val_loss = history_dict['val_loss']

# epochs = range(1, len(acc) + 1)

# # "bo" is for "blue dot"
# plt.plot(epochs, loss, 'bo', label='Training loss')
# # b is for "solid blue line"
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()

# plt.show()

export_model = tf.keras.Sequential([
  vectorize_layer,
  model,
  layers.Activation('sigmoid')
])

export_model.compile(
    loss=losses.SparseCategoricalCrossentropy(from_logits=False), optimizer="adam", metrics=['accuracy']
)

# Test it with `raw_test_ds`, which yields raw strings
loss, accuracy = export_model.evaluate(raw_val_ds)
print(accuracy)


examples = [
  "Focus on what you can do, not what you would like to do.",
  "Better to keep quiet and be thought a fool, than open your mouth and remove all doubt.",
  "Space, the final frontier... after cotton candy.",
  "Good news, bad news, who can say? In the end, we are dust.",
  "I was drawn to all the wrong things: I liked to drink, I was lazy, I didnt have a god, politics, ideas, ideals. I was settled into nothingness; a kind of non-being, and I accepted it. I didnt make for an interesting person. I didnt want to be interesting, it was too hard. What I really wanted was only a soft, hazy space to live in, and to be left alone. On the other hand, when I got drunk I screamed, went crazy, got all out of hand. One kind of behavior didnt fit the other. I didnt care.",
  'I dont think any of the early Romantic composers knew how to write for the piano... The music of that era is full of empty theatrical gestures, full of exhibitionism, and it has a worldly, hedonistic quality that simply turns me off.',
  'As a kid, I watched Bugs Bunny cartoons, and for some reason Pepe Le Pew, the indomitable French skunk pursuing his would-be kitty paramour, left his mark on me: became an instant emblem of odoriferous hubris, hedonistic bad behavior. He was an entry-level Dominique Strauss-Kahn a rookie Marquis de Sade',
  'If you want the present to be different from the past, study the past.',
  'All things excellent are as difficult as they are rare.',
  'Do not weep; do not wax indignant. Understand.',
  "The will to live guarantees survival.",
  "He who has a why to live can bear almost any how."
]

print(export_model.predict(examples))

print("Label 0 corresponds to", raw_train_ds.class_names[0])
print("Label 1 corresponds to", raw_train_ds.class_names[1])
print("Label 2 corresponds to", raw_train_ds.class_names[2])