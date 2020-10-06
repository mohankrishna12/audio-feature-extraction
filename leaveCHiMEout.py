import pandas as pd
import numpy as np
from numpy import loadtxt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import QuantileTransformer
from scipy import stats
import numpy as np
import essentia
from sklearn.gaussian_process import GaussianProcessClassifier
from essentia.standard import *
import essentia.standard as es
import scipy
from sklearn.neighbors import KNeighborsClassifier
import sklearn



# load the dataset
dataset = pd.read_csv("/home/nick/Documents/audio_features/finalmodel16bitwithgroups.csv")

from pathlib import Path

model_string = Path("/home/nick/audio-feature-extraction/ANOVA_KNN_K Nearest Neighbors_features.txt").read_text()
model_string = model_string.replace('\n', '')
model_features = model_string.split("&&")
#print(dataset)
X = dataset[model_features]
X=X.fillna(0)

X = X.to_numpy()
d = dataset.to_numpy()
#normally 628
#print(dataset.shape)
print(X.shape)

#X = dataset[:,2:152]

y = d[:,2]
y=y.astype('int')

group = d[:,1]
group=group.astype('int')

seed = 8
# prepare models
models = []

models.append(('KNN1', KNeighborsClassifier(n_neighbors=3, weights='distance')))



results_df = pd.DataFrame(columns = ['Name of File', 'Actual', 'Predicted', 'Correct?', 'Avg. Distance', '3 Nearest Neighbors'])


for name, model in models:

	#print(len(y))
	for i in range(0,len(y)):
		#print(group[i])
		#print(group)
		print(i)
		currname = d[:,0][i]
		#print(currname)
		currGroup = group[i]
		if(currGroup<18 or currGroup>23):
			continue
		testPoint = X[i]
		testY = y[i]

		#currPoints = dataset[dataset['group']!=currGroup]
		currPoints = dataset.loc[(dataset['group']<18 ) | (dataset['group']>23)]
		currY = currPoints.to_numpy()[:,2].astype('int')
		#for y in currY:
		#	print(currY)

		currX = currPoints[model_features]

		currX=currX.fillna(0)
		currX=currX.to_numpy()
		#print(currX)
		print(currX.shape)
		qt = QuantileTransformer(output_distribution='uniform')
		currX = qt.fit_transform(currX)
		#print(Xnew)
		#Xnew = qt.fit_transform(X)
		

		model.fit(currX,currY)

		#print(testPoint.shape)
		#print(testPoint.reshape(1,-1))

		prediction = model.predict(qt.transform(testPoint.reshape(1,-259)))

		neigh = model.kneighbors(qt.transform(testPoint.reshape(1,-259)))
		#print(neigh)
		#print(neigh[0])

		#print("NUM NEIGHBORS: " + str(len(neigh[1][0])))
					#print("INTERACTIVE NEIGHBORS: " + str(intneighs))
					#print("NON-INTERACTIVE NEIGHBORS: " + str(nonneighs))
					
		import statistics
		names = ''
		for file in neigh[1][0]:
			names+= currPoints.to_numpy()[:,0][file] + ", "

		print(names)

		#print(names)
		distance = statistics.mean(neigh[0][0])
		#print("AVG DISTANCE: " + str(distance))
		#print("Test for Value: " +str(i))
		#print("Actual Value: " + str(testY))
		#print("Predicted Value: " + str(prediction))


		results_df = results_df.append({'Name of File': currname, 'Actual' : testY, 'Predicted' : prediction[0], 'Correct?': int(testY)==int(prediction[0]), 'Avg. Distance' : str(distance), '3 Nearest Neighbors': names}, ignore_index=True) 

results_df.to_csv("LeaveCHIMEOutCV2.csv")
