import numpy as np
import pandas as pd
 
np.random.seed(1)
csv_file_url = 'data/raccoon_labels.csv'
full_data = pd.read_csv(csv_file_url)
total_file_number = len(full_data)
print("There are total {} examples in this dataset.".format(total_file_number))
full_data.head() #Viewing the first 5 lines


num_train = int(total_file_number*0.85)
num_validation = int(total_file_number*0.1)
num_test = total_file_number-num_train-num_validation

assert num_train + num_validation + num_test <= total_file_number, "Not enough examples for your choice."
print("Looks good! {} for train, {} for validation and {} for test.".format(num_train, num_validation, num_test))


index_train = np.random.choice(total_file_number, size=num_train, replace=False)
index_validation_test = np.setdiff1d(list(range(total_file_number)), index_train)
index_validation = np.random.choice(index_validation_test, size=num_validation, replace=False)
index_test = np.setdiff1d(index_validation_test, index_validation)

train = full_data.iloc[index_train]
validation = full_data.iloc[index_validation]
test = full_data.iloc[index_test]


train.to_csv('data/data_train.csv', index=None)
validation.to_csv("data/data_validation.csv", index=None)
test.to_csv('data/data_test.csv', index=None)

print("All done!")