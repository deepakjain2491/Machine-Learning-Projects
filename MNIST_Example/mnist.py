import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds


#Loading the data
mnist_data, mnist_info = tfds.load('mnist', with_info = True, as_supervised = True)


# Preprocessing the data
mnist_train, mnist_test = mnist_data['train'], mnist_data['test']

validation_sample = 0.1 * mnist_info.splits['train'].num_examples
validation_sample = tf.cast(validation_sample, tf.int64)

test_sample = mnist_info.splits['test'].num_examples
test_sample = tf.cast(test_sample, tf.int64)

def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255.
    return image, label

scaled_train_and_validation_data = mnist_train.map(scale)
scaled_test_data = mnist_test.map(scale)


BUFFER_SIZE = 10000
shuffled_train_and_validation_data = scaled_train_and_validation_data.shuffle(BUFFER_SIZE)

validation_data = shuffled_train_and_validation_data.take(validation_sample)
train_data = shuffled_train_and_validation_data.skip(validation_sample)

BATCH_SIZE = 150
train_data = train_data.batch(BATCH_SIZE)
validation_data = validation_data.batch(validation_sample)
test_data = scaled_test_data.batch(test_sample)

validation_inputs, validation_targets = next(iter(validation_data))

validation_inputs, validation_targets = next(iter(validation_data))


# The Model
input_size = 784
output_size = 10
hidden_layer_size = 150

model = tf.keras.Sequential([
                            tf.keras.layers.Flatten(input_shape=(28,28,1)),
                            tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
                            tf.keras.layers.Dense(hidden_layer_size, activation='sigmoid'),
                            tf.keras.layers.Dense(output_size, activation = 'softmax')
                            ])

# Compling the model
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

# Training the model
NUM_EPOCHS = 10

model.fit(train_data, epochs = NUM_EPOCHS, validation_data =(validation_inputs, validation_targets), verbose=2)

# Testing the model
test_loss , test_accuracy =  model.evaluate(test_data)
print(test_accuracy, test_loss)

