import numpy as np
import tensorflow as tf

npz = np.load('D:/Python_projects/Section_51/Audiobooks_train_data.npz')
train_inputs = npz['input'].astype(np.float)
train_targets = npz['targets'].astype(np.int)


npz = np.load('D:/Python_projects/Section_51/Audiobooks_validation_data.npz')
validation_inputs = npz['input'].astype(np.float)
validation_targets = npz['targets'].astype(np.int)

npz = np.load('D:/Python_projects/Section_51/Audiobooks_test_data.npz')
test_inputs = npz['input'].astype(np.float)
test_targets = npz['targets'].astype(np.int)


# CREATING THE MODEL

input_size = 10
output_size = 2
hidden_layers_size = 150

model = tf.keras.Sequential([
                            tf.keras.layers.Dense(hidden_layers_size, activation= 'relu'),
                            tf.keras.layers.Dense(hidden_layers_size, activation='relu'),
                            tf.keras.layers.Dense(output_size, activation = 'softmax')
                            ])

model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])

early_stopping = tf.keras.callbacks.EarlyStopping(patience=2)
BATCH_SIZE = 100
MAX_EPOCHS = 100

model.fit(train_inputs,
          train_targets,
          batch_size=BATCH_SIZE,
          callbacks= [early_stopping],
          epochs=MAX_EPOCHS,
          validation_data=(validation_inputs, validation_targets),
          verbose=2)


# Testing the model
test_loss , test_accuracy = model.evaluate(test_inputs, test_targets)
print(test_accuracy, test_loss)
