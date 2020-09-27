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
import sounddevice as sd
from scipy.io.wavfile import write
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import time
import essentia
from essentia.standard import *
import essentia.standard as es
import pandas as pd
import scipy
import sklearn
# librosa
import librosa
import librosa.display
import audio_extractors
import anki_vector


# load the dataset
#dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')
#dataset = loadtxt('all_features.csv', delimiter=',')
#dataset = pd.read_csv("FINAL.csv")
dataset = pd.read_csv("emmanuelMODEL.csv")
'''
podcast = pd.read_csv("podcasttest.csv")
newpodcast = pd.read_csv("podcastnewtest.csv")
s18 = pd.read_csv("chimes18test.csv")
s23 = pd.read_csv("chimes23test.csv")
s24 = pd.read_csv("s24.csv")
friends = pd.read_csv("friendsRECORDED.csv")
friends = friends.drop([friends.columns[0]], axis=1)
sports = pd.read_csv("MCvARS_RECORDED.csv")
sports = sports.drop([sports.columns[0]], axis=1)
newone = pd.read_csv("S23U02CH3RecordedFromReplayUntrained.csv")
newone = pd.read_csv("S23U02CH3RecordedFromReplayUntrained.csv")
s3u3 = pd.read_csv("s3u3RECORDEDchopped.csv")
#s3u3 = s3u3.drop([newone.columns[0], newone.columns[1]], axis=1)
s3u2 = pd.read_csv("s3u1RECORDEDchopped.csv")
#s3u1 = s3u1.drop([newone.columns[0], newone.columns[1]], axis=1)
avche = pd.read_csv("AVvCHE_RECORDED_chopped.csv")
#sports = pd.read_csv("sportstest.csv")
me = pd.read_csv("ME_TALKING_PLUSSILENCE_CHOPPED.csv")
s4u1 = pd.read_csv("s4u1RECORDEDchopped.csv")
rebyt = pd.read_csv("youtube-5s-recordings.csv")
JOE = pd.read_csv("JOHNOLIVER_ERA_RECORDED_CHOPPED.csv")
mept2 = pd.read_csv("METALKINGRECORDEDPT2_CHOPPED.csv")
friends16 = pd.read_csv("FRIENDS16BIT_RECORDED_CHOPPED.csv")
s4u2 = pd.read_csv("s4u2RECORDEDchopped.csv")
mept3 = pd.read_csv("meRECORDEDpt3.csv")
direct = pd.read_csv("directRecording.csv")
readtest = pd.read_csv("3minutereadRECORDED.csv")
'''

#dataset = dataset.drop([dataset.columns[0], dataset.columns[1]], axis=1)

dataset = dataset.to_numpy()
#normally 628
X = dataset[:,1:151]
y = dataset[:,0]


from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from joblib import dump,load
import os
import pandas as pd




seed = 8
# prepare models
models = []

#models.append(('KNN1', KNeighborsClassifier(n_neighbors=3, weights='distance')))
#models.append(('KNN2', KNeighborsClassifier(weights='distance')))
#models.append(('KNN3', KNeighborsClassifier()))
models.append(('KNN4', KNeighborsClassifier(n_neighbors=30, weights='distance')))
#models.append(('KNN4', KNeighborsClassifier(n_neighbors=7, weights='distance')))

bigTests=True
if bigTests:
	# Capture audio from through microphone
	##############################################
	fs = 44100  # Sample rate
	seconds = 5  # Duration of recording
	directory = '/home/nick/Documents/audio_features/'
	print("If you want to record enter y")
	folder = 'LIVETEST2/'
	record = input()
	if(record == 'y'):
		
		
		
		print("press enter to record")
		input()
		audioRecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2, dtype='int16')
		
		print("recording...")
		sd.wait()  # Wait until recording is finished
		write(directory + 'LongAudio/longOutputAudio.wav', fs, audioRecording)

		print("Waiting 300 milliseconds for audio to be saved and chunked up...")
		time.sleep(.300)
		# print("Waiting 2 seconds while snippets are chunked up...")
		# time.sleep(3)
		##############################################


		# Creating audio snippets

		from pydub import AudioSegment
		from pydub.utils import make_chunks
		from datetime import datetime

		myaudio = AudioSegment.from_file(directory + 'LongAudio/longOutputAudio.wav', "wav")
		samplerate = 44100
		myaudio = myaudio.set_frame_rate(samplerate)
		# myaudio, sr = librosa.load(filename, sr=8000)
		chunk_length_ms = 5000
		chunks = make_chunks(myaudio, chunk_length_ms)

		for i, chunk in enumerate(chunks):
			#now = datetime.now()
			chunk_name = 'livetest' + "{0}.wav".format(i)
			print("Exporting...", chunk_name)
			chunk.export(directory + folder + chunk_name, format="wav")

		print("writing files")
		time.sleep(3.0)


	
	# evaluate each model in turn
	results = []
	names = []
	scoring = 'accuracy'
	for name, model in models:
		kfold = model_selection.KFold(n_splits=5, random_state=8)
		
		qt = QuantileTransformer()
		Xnew = qt.fit_transform(X)

		interactive=0
		nonint = 0
		for i in y:
			if(i==0.):
				interactive+=1
			else:
				nonint+=1

		
		print("INTERACTIVE TRAINING: " + str(interactive))
		print("NON-INTERACTIVE TRAINING: " + str(nonint))
		
		model.fit(Xnew,y)

		#model.fit(X,y)
		#print(model)
		#model =  load('ourKNN.joblib')
		cv_results = model_selection.cross_val_score(model, Xnew, y, cv=kfold, scoring=scoring)
		results.append(cv_results)
		names.append(name)
		msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
		print(name)
		print(msg)
		
		
		silent = 0
		interactive = 0
		nonint = 0
		for filename in os.listdir(directory + folder):
			if filename.endswith(".wav") or filename.endswith(".mp3"):
				print(folder+filename)
				allFeatures = audio_extractors.getFeatures(folder+filename)

			    

				if(isinstance(allFeatures,str)):
				  #print("------------------------------------------------------\n\n\n\n\n\n\n\nSILENT\n\n\n\n\n\n\n\n------------------------------------------------------")
					silent+=1
				else:
					#normData = allFeatures.to_numpy()[:,0:627]
					normData = qt.transform(allFeatures.to_numpy()[:,2:152])
					#print(normData)
					ourPrediction = model.predict(normData)
					print(ourPrediction)
					#print(model[0])
					
					neigh = model.kneighbors(normData)
					'''
					#print(neigh.shape)
					print("----------HERE------------")
					#print(allFeatures.to_numpy()[:,0:10])
					#print(neigh)
					#print(neigh[0])
					#print(neigh[1])
					intneighs = 0
					nonneighs = 0
					#INTINDEX = 1783
					INTINDEX = 341
					for i in neigh[1][0]:
						if i<INTINDEX or (i>=929 and i<=1003):
							intneighs+=1
						else:
							nonneighs+=1

						#print(X[i])
						#print(y[i])
					print("NUM NEIGHBORS: " + str(len(neigh[1][0])))
					print("INTERACTIVE NEIGHBORS: " + str(intneighs))
					print("NON-INTERACTIVE NEIGHBORS: " + str(nonneighs))
					'''
					import statistics
					print("AVG DISTANCE: " + str(statistics.mean(neigh[0][0])))

					ourPrediction = int(ourPrediction[0])
					print(ourPrediction)
					if(ourPrediction==1):
						nonint+=1
						print("------------------------------------------------------\nNOT INTERACTIVE\n------------------------------------------------------")
					else:
						interactive+=1
						print("------------------------------------------------------\nINTERACTIVE\n--------------------------------------------------")

		print("-------------just recorded------------")
		print("*************************         " + name + "           ****************************")
		print("INTERACTIVE: " + str(interactive))
		print("NON-INTERACTIVE: " + str(nonint))
		print("SILENT: " + str(silent))
		print("*******************************************************************************")
		

		
	
	'''
	silent = 0
	interactive = 0
	nonint = 0
	normData = qt.transform(me.to_numpy()[:,2:152])
	predictions = model.predict(normData)
	print(predictions)
	for i in predictions:
		if(i==0.):
			interactive+=1
		else:
			nonint+=1

	
	print("INTERACTIVE: " + str(interactive))
	print("NON-INTERACTIVE: " + str(nonint))
	print("SILENT: " + str(silent))
	print("-----------------------------------------------------------")
	
	


	print("-------------ME PT2---------------")

	silent = 0
	interactive = 0
	nonint = 0
	normData = qt.transform(mept2.to_numpy()[:,2:152])
	predictions = model.predict(normData)
	print(predictions)
	for i in predictions:
		if(i==0.):
			interactive+=1
		else:
			nonint+=1

	
	print("INTERACTIVE: " + str(interactive))
	print("NON-INTERACTIVE: " + str(nonint))
	print("SILENT: " + str(silent))


	print("-------------rebecca john oliver---------------")
	silent = 0
	interactive = 0
	nonint = 0
	normData = qt.transform(rebyt.to_numpy()[:,2:152])
	predictions = model.predict(normData)
	print(predictions)
	for i in predictions:
		if(i==0.):
			interactive+=1
		else:
			nonint+=1

	
	print("INTERACTIVE: " + str(interactive))
	print("NON-INTERACTIVE: " + str(nonint))
	print("SILENT: " + str(silent))

	print("-------------me john oliver---------------")
	silent = 0
	interactive = 0
	nonint = 0
	normData = qt.transform(JOE.to_numpy()[:,2:152])
	predictions = model.predict(normData)
	print(predictions)
	for i in predictions:
		if(i==0.):
			interactive+=1
		else:
			nonint+=1

	
	print("INTERACTIVE: " + str(interactive))
	print("NON-INTERACTIVE: " + str(nonint))
	print("SILENT: " + str(silent))

	print("-------------friends 16---------------")
	silent = 0
	interactive = 0
	nonint = 0
	normData = qt.transform(friends16.to_numpy()[:,2:152])
	predictions = model.predict(normData)
	print(predictions)
	for i in predictions:
		if(i==0.):
			interactive+=1
		else:
			nonint+=1

	
	print("INTERACTIVE: " + str(interactive))
	print("NON-INTERACTIVE: " + str(nonint))
	print("SILENT: " + str(silent))

	print("-------------friends 32---------------")
	silent = 0
	interactive = 0
	nonint = 0
	normData = qt.transform(friends.to_numpy()[:,2:152])
	predictions = model.predict(normData)
	print(predictions)
	for i in predictions:
		if(i==0.):
			interactive+=1
		else:
			nonint+=1

	
	print("INTERACTIVE: " + str(interactive))
	print("NON-INTERACTIVE: " + str(nonint))
	print("SILENT: " + str(silent))

	print("-------------s4u2---------------")
	silent = 0
	interactive = 0
	nonint = 0
	normData = qt.transform(s4u2.to_numpy()[:,2:152])
	predictions = model.predict(normData)
	print(predictions)
	for i in predictions:
		if(i==0.):
			interactive+=1
		else:
			nonint+=1

	
	print("INTERACTIVE: " + str(interactive))
	print("NON-INTERACTIVE: " + str(nonint))
	print("SILENT: " + str(silent))

	print("-------------mept3---------------")
	silent = 0
	interactive = 0
	nonint = 0
	normData = qt.transform(mept3.to_numpy()[:,2:152])
	predictions = model.predict(normData)
	print(predictions)
	for i in predictions:
		if(i==0.):
			interactive+=1
		else:
			nonint+=1

	
	print("INTERACTIVE: " + str(interactive))
	print("NON-INTERACTIVE: " + str(nonint))
	print("SILENT: " + str(silent))

	print("-------------direct---------------")
	silent = 0
	interactive = 0
	nonint = 0
	normData = qt.transform(direct.to_numpy()[:,2:152])
	predictions = model.predict(normData)
	print(predictions)
	for i in predictions:
		if(i==0.):
			interactive+=1
		else:
			nonint+=1

	
	print("INTERACTIVE: " + str(interactive))
	print("NON-INTERACTIVE: " + str(nonint))
	print("SILENT: " + str(silent))


	print("-------------3 min read---------------")
	silent = 0
	interactive = 0
	nonint = 0
	normData = qt.transform(readtest.to_numpy()[:,2:152])
	predictions = model.predict(normData)
	print(predictions)
	for i in predictions:
		if(i==0.):
			interactive+=1
		else:
			nonint+=1

	
	print("INTERACTIVE: " + str(interactive))
	print("NON-INTERACTIVE: " + str(nonint))
	print("SILENT: " + str(silent))

    		
	
	

	'''
