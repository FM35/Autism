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