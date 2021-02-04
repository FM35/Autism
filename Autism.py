import numpy as np
from sklearn import preprocessing
from scipy.io import arff
import pandas as pd

data = arff.loadarff('Autism_Screening_Adult/Autism-Adult-Data.arff')
Autism_unprocessed = pd.DataFrame(data[0])

inputs = Autism_unprocessed.drop(['age', 'result'], axis = 1)

for x in inputs:
    Autism_unprocessed[x] = inputs[x].str.decode('utf-8')

Autism_unprocessed = Autism_unprocessed.drop(['ethnicity','relation', 'age_desc', 'used_app_before'], axis = 1)
Autism_unprocessed = Autism_unprocessed.dropna(axis = 0)
Autism_unprocessed = Autism_unprocessed[Autism_unprocessed.age != 383]
Autism_unprocessed['Class/ASD'] = Autism_unprocessed['Class/ASD'].map({'YES': 1,'NO': 0})
Autism_unprocessed['jundice'] = Autism_unprocessed['jundice'].map({'yes': 1,'no': 0})
Autism_unprocessed['austim'] = Autism_unprocessed['austim'].map({'yes': 1,'no': 0})
Autism_unprocessed['gender'] = Autism_unprocessed['gender'].map({'m': 1,'f': 0})

Autism_unprocessed = pd.get_dummies(Autism_unprocessed, drop_first = True)
Autism_unprocessed = Autism_unprocessed.reset_index(drop = True)

#Shuffling Data
Autism_unprocessed = Autism_unprocessed.sample(frac=1)
Autism_unprocessed = Autism_unprocessed.reset_index(drop = True)

unscaled_inputs = Autism_unprocessed.drop(['Class/ASD'], axis = 1)
targets = Autism_unprocessed['Class/ASD']
#print(targets)

num_one_targets = int(np.sum(targets))
zero_targets_counter = 0
indices_to_remove = []

for i in range(targets.shape[0]):
    if targets[i] ==0:
        zero_targets_counter += 1
        if zero_targets_counter > num_one_targets:
            indices_to_remove.append(i)

unscaled_inputs_equal_priors = unscaled_inputs.drop(indices_to_remove)
unscaled_inputs_equal_priors = unscaled_inputs_equal_priors.reset_index(drop = True)
targets_equal_priors = targets.drop(indices_to_remove)
targets_equal_priors = targets_equal_priors.reset_index(drop = True)

unscaled_inputs_equal_priors['targets'] = targets_equal_priors

unscaled_inputs_equal_priors = unscaled_inputs_equal_priors.sample(frac=1)

targets_equal_priors = unscaled_inputs_equal_priors['targets']
unscaled_inputs_equal_priors = unscaled_inputs_equal_priors.drop(['targets'], axis = 1)

unscaled_inputs_equal_priors = unscaled_inputs_equal_priors.reset_index(drop = True)

#Feature Scaling the inputs
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(unscaled_inputs_equal_priors)
scaled_inputs_equal_priors = scaler.transform(unscaled_inputs_equal_priors)

samples_count = scaled_inputs_equal_priors.shape[0]

train_samples_count = int(0.8*samples_count)
validation_samples_count = int(0.1*samples_count)
test_samples_count = samples_count - train_samples_count - validation_samples_count

train_inputs = scaled_inputs_equal_priors[:train_samples_count]
train_targets = targets_equal_priors[:train_samples_count]

#print(np.sum(train_targets)/ train_samples_count)

validation_inputs = scaled_inputs_equal_priors[train_samples_count:train_samples_count+validation_samples_count]
validation_targets = targets_equal_priors[train_samples_count:train_samples_count+validation_samples_count]

#print(np.sum(validation_targets)/ validation_samples_count)

test_inputs = scaled_inputs_equal_priors[train_samples_count+validation_samples_count:]
test_targets = targets_equal_priors[train_samples_count+validation_samples_count:]

#print(np.sum(test_targets)/ test_samples_count)
print(test_targets.value_counts())

print(unscaled_inputs_equal_priors.describe(include = 'all'))

#np.savez('Autism_data_train', inputs = train_inputs, targets = train_targets)
#np.savez('Autism_data_validation', inputs = validation_inputs, targets = validation_targets)
#np.savez('Autism_data_test', inputs = test_inputs, targets = test_targets)