import numpy as np
from sklearn import preprocessing
from scipy.io import arff
import pandas as pd

data = arff.loadarff('Autism_Screening_Adult/Autism-Adult-Data.arff')
Autism_unprocessed = pd.DataFrame(data[0])

inputs = Autism_unprocessed.drop(['age', 'result'], axis = 1)

for x in inputs:
    Autism_unprocessed[x] = inputs[x].str.decode('utf-8')

Autism_unprocessed = Autism_unprocessed.drop(['ethnicity','relation', 'age_desc'], axis = 1)
Autism_unprocessed = Autism_unprocessed.dropna(axis = 0)
Autism_unprocessed = Autism_unprocessed[Autism_unprocessed.age != 383]
Autism_unprocessed['Class/ASD'] = Autism_unprocessed['Class/ASD'].map({'YES': 1,'NO': 0})

Autism_unprocessed = Autism_unprocessed.reset_index(drop = True)
unscaled_inputs = Autism_unprocessed.drop(['Class/ASD'], axis = 1)
targets = Autism_unprocessed['Class/ASD']

print(Autism_unprocessed['used_app_before'].value_counts())


num_one_targets = int(np.sum(targets))
zero_targets_counter = 0
indices_to_remove = []

for i in range(targets.shape[0]):
    if targets[i] ==0:
        zero_targets_counter += 1
        if zero_targets_counter > num_one_targets:
            indices_to_remove.append(i)

unscaled_inputs_equal_priors = unscaled_inputs.drop(indices_to_remove)
targets_equal_priors = targets.drop(indices_to_remove)