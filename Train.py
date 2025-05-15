import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

data_dict = pickle.load(open('./data.pickle', 'rb'))
# print(data_dict.keys())
# print(data_dict)
expected_length = 42 
filtered_data = [x for x in data_dict['data'] if len(x) == expected_length]
filtered_labels = [label for x, label in zip(data_dict['data'], data_dict['labels']) if len(x) == expected_length]

data = np.asarray(filtered_data)
labels = np.asarray(filtered_labels)

# print(data)
# print(labels)

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size = 0.2, shuffle = True, stratify = labels)
model = RandomForestClassifier()

model.fit(x_train, y_train)
y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)
print(score)

f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()