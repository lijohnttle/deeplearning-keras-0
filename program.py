from keras.models import Sequential
from keras.layers import Dense
from pandas import read_csv
import numpy

seed = 13
numpy.random.seed(seed)

# import dataset
filename = 'data.csv'
dataframe = read_csv(filename)

# split output variables
array = dataframe.values

X = array[:, 0:11]
Y = array[:, 11]
dataframe.head()

# build the model
model = Sequential()
model.add(Dense(11, input_dim=11, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the model
model.fit(X, Y, epochs=200, batch_size=10)

# score the model
scores = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
