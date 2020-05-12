import numpy as np
from sklearn import preprocessing
import pandas as pd

# Extarcting data from CSV File
raw_csv_data = np.loadtxt('D:/OneDrive - Concordia University - Canada/Attachments/Data_Science_Projects/S51_L353/Audiobooks_data.csv', delimiter=',')

unscaled_raw_data = raw_csv_data[:,1:-1]
targets_all = raw_csv_data[:,-1]


#Balancing the dataset
num_one_targets = int(np.sum(targets_all))
zero_taget_counter = 0
indices_to_remove = []

for i in range(targets_all.shape[0]):
    if targets_all[i] == 0:
        zero_taget_counter += 1
        if zero_taget_counter > num_one_targets:
            indices_to_remove.append(i)

unscaled_raw_data_equal_priors = np.delete(unscaled_raw_data, indices_to_remove, axis=0)
targets_all_equal_priors = np.delete(targets_all, indices_to_remove, axis=0)

## Standardizing the inputs
scaled_inputs = preprocessing.scale(unscaled_raw_data_equal_priors)

# Shuffling the data
shuffled_indices = np.arange(scaled_inputs.shape[0])
np.random.shuffle(shuffled_indices)

shuffled_inputs = scaled_inputs[shuffled_indices]
shuffled_targets = targets_all_equal_priors[shuffled_indices]

## CREATING TRAINING, VALIDATION AND TESTING DATA
sample_count = shuffled_inputs.shape[0]

train_samples = int(0.8* sample_count)
validation_samples = int(0.1 * sample_count)
test_samples = train_samples - validation_samples

# Extracting TRAIN, VALIDATION AND TEST DATASETS
train_inputs = shuffled_inputs[:train_samples]
train_targets =shuffled_targets[:train_samples]

validation_inputs = shuffled_inputs[train_samples:train_samples + validation_samples]
validation_targets = shuffled_targets[train_samples:train_samples + validation_samples]

test_inputs = shuffled_inputs[train_samples + validation_samples:]
test_targets = shuffled_targets[train_samples + validation_samples:]


## Saving the data
np.savez('Audiobooks_train_data', input= train_inputs, targets = train_targets)
np.savez('Audiobooks_validation_data', input= validation_inputs, targets = validation_targets)
np.savez('Audiobooks_test_data', input= test_inputs, targets = test_targets)
