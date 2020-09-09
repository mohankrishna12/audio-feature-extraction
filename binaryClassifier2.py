# first neural network with keras make predictions
print("Hi")
import pandas as pd
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
# load the dataset
#dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')
#dataset = loadtxt('all_features.csv', delimiter=',')
dataset = pd.read_csv("all_features.csv")
dataset = dataset.drop([dataset.columns[0], dataset.columns[1]], axis=1)
#dataset = dataset.iloc[1:]


# split into input (X) and output (y) variables
#print(dataset.shape)
#print(dataset)
dataset = dataset.to_numpy()
#print(dataset.shape)
#print(dataset)

X = dataset[:,1:673]
y = dataset[:,0]
print(X.shape)
print(X)
print(y.shape)
print(y)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10)


# define the keras model
#def create_baseline():
model = Sequential()
#model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(512, input_dim=670, activation='relu'))
#model.add(Dense(128, activation='relu'))
#model.add(Dense(32, activation='relu'))
#model.add(Dense(2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#return model 


#print(X_train.shape)
#print(y_train.shape)
#print(X_test.shape)
#print(y_test)
# fit the keras model on the dataset
#model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=20)


model.fit(X_train, y_train, epochs=10)

loss, accuracy = model.evaluate(X_test, y_test)
print("TEST ACCURACY %.2f" % (accuracy*100))



#print(X_test.shape)
blah = X_test[0:5]
print(blah.shape)
print(model.predict(blah))
print(y_test[0:5])

#estimator = KerasClassifier(build_fn=create_baseline, epochs=10, batch_size=32, verbose=2)
#kfold = StratifiedKFold(n_splits=10, shuffle=True)
#results = cross_val_score(estimator, X, y, cv=kfold)
#print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))





# make class predictions with the model
#predictions = model.predict_classes(X)
# summarize the first 30 cases
#for i in range(30):
#	print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))
# Save my model

#model.save('interactivity_model_150epochs.h5')
