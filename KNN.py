# first neural network with keras make predictions
print("Hi")
import pandas as pd
from sklearn.svm import LinearSVC
import numpy as np
from joblib import dump,load
from numpy import loadtxt
#from keras.models import Sequential
#from keras.layers import Dense
#rom keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import confusion_matrix
from scipy import stats

# load the dataset
#dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')
#dataset = loadtxt('all_features.csv', delimiter=',')
dataset = pd.read_csv("morechimeandtv.csv")
podcast = pd.read_csv("podcasttest.csv")
#s18 = pd.read_csv("chimes18test.csv")
#s23 = pd.read_csv("chimes23test.csv")



dataset = dataset.drop([dataset.columns[0], dataset.columns[1]], axis=1)

#print(dataset.shape)
#dataset =np.abs(stats.zscore(dataset))
#outliers = np.where(dataset>3)
#dataset[(np.abs(stats.zscore(dataset)) < 3).all(axis=1)] 

#print(dataset.shape)
#dataset = dataset.iloc[1:]
#print(dataset.columns[374])
# split into input (X) and output (y) variables
#print(dataset.shape)
#print(dataset)
dataset = dataset.to_numpy()
#print(dataset.shape)
#print(dataset)

X = dataset[:,1:628]
y = dataset[:,0]


#print(X.shape)
#print(X)
#print(y.shape)
#print(y)

#X = pd.DataFrame(X)


#scaler = QuantileTransformer()
#X = scaler.fit_transform(X)
#print(X)

#z = np.abs(stats.zscore(X))

#print(z)
#print(z.shape)

#outliers = np.where(z>3)

#print(z[0][192])
#print(len(outliers[1]))
#print(len(outliers[0]))
#print(len(outliers[0])/(2771*627))
#print(outliers)

#print(2771*690)
#print(z[2064][374])
#print(z[2238][374])
#print(z[2240][374])
#print(z[2487][374])
#print('here')
#print(z)
#filtered =  (z<3).all(axis=1)
#print(filtered)
#print(filtered[2064])
#print(filtered[2065])
#X_new = X[(z<3).all(axis=1)]
#print(X_new.shape)




#scaler = StandardScaler()
#X =scaler.fit(X)
#print(scaler)
#StandardScaler()


interactive = 0
non = 0

for i in y:
	if i==0.0:
		non +=1
	else:
		interactive+=1

print("Interactive: " + str(interactive))
print("Non-Interactive: " + str(non))


#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=np.random.randint(100))
#from sklearn.decomposition import PCA



#clf = make_pipeline(QuantileTransformer(), svm.SVC(gamma='auto'))

#scores = cross_val_score(clf, X, y, cv=5)
#print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
#clf = make_pipeline(StandardScaler(), LinearSVC(random_state=0, tol=1e-7, max_iter=50000))


#clf.fit(X_train, y_train)

#print(clf.named_steps['linearsvc'].coef_)

#print(clf)
#blah = X_test[0:50]
#print(blah.shape)
#predictions = clf.predict(X_test)
#print(predictions)
#print(y_test)

#print(confusion_matrix(y_test, predictions))

#print('------------SVM accuracy---------------')
#accuracy = clf.score(X_test,y_test)
#print("TEST ACCURACY %.2f" % (accuracy*100))
#scores = cross_val_score(clf, X, y, cv=5)
#print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean()*100, scores.std() * 2))

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.decomposition import PCA



seed = 8
# prepare models
models = []
#models.append(('LR', LogisticRegression()))
#models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
#models.append(('CART', DecisionTreeClassifier()))
#models.append(('NB', GaussianNB()))
#models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
for name, model in models:
	kfold = model_selection.KFold(n_splits=5, random_state=seed)
	model = make_pipeline(QuantileTransformer(), model)

	model.fit(X,y)
	cv_results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)



	print("-------------podcast (hasnt seen before)------------")
	predictions = model.predict(podcast.to_numpy()[:,0:627])
	right = 0
	wrong = 0
	for i in predictions:
		if(i==0.):
			right+=1
		else:
			wrong+=1

	print("right: " + str(right))
	print("wrong: " + str(wrong))

	print(predictions)
'''
	print("-------------s18 (seen before)------------")
	predictions = model.predict(s18.to_numpy()[:,0:627])
	right = 0
	wrong = 0
	for i in predictions:
		if(i==0.):
			wrong+=1
		else:
			right+=1

	print("right: " + str(right))
	print("wrong: " + str(wrong))

	print(predictions)

	print("-------------s23 (hasnt seen before)------------")
	predictions = model.predict(s23.to_numpy()[:,0:627])
	right = 0
	wrong = 0
	for i in predictions:
		if(i==0.):
			wrong+=1
		else:
			right+=1

	print("right: " + str(right))
	print("wrong: " + str(wrong))

	print(predictions)
'''

#from joblib import dump,load

#dump(clf, 'ourPCASVM.joblib')

'''



















#print(y_test.shape)




# define the keras model
#def create_baseline():
model = Sequential()
#model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(512, input_dim=670, activation='relu'))
#model.add(Dense(128, activation='relu'))
#model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#	return model 


#print(X_train.shape)
#print(y_train.shape)
#print(X_test.shape)
#print(y_test)
# fit the keras model on the dataset
#model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=20)

model.fit(X_train, y_train, epochs=10, batch_size=4)

loss, accuracy = model.evaluate(X_test, y_test)
print("TEST ACCURACY %.2f" % (accuracy*100))


#print(X_test.shape)
blah = X_test
#print(blah.shape)
preds = model.predict(blah)

print(preds)
#print(confusion_matrix(y_test, preds))


print("---------------------------SVM NEXT-----------------------------------------")


#model = svm.SVC()
#model.fit(X_train,y_train)




'''








'''
clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=5)
clf.fit(X, y)
print('------------SGD accuracy---------------')
print(clf.score(X_test,y_test)*100)
'''


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
#testset = pd.read_csv("test.csv")

#preds = model.predict(testset)
#print(preds)

#model.save('testNetepochs.h5')
