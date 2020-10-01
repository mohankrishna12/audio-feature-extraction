print("Aloha")
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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import QuantileTransformer
# librosa
import librosa
import librosa.display
import audio_extractors
import anki_vector

s = "0030431E"
ipaddr = "10.0.0.120"
name = "Vector-N5H2"
print('-------------------------------------------------------')
dataset = pd.read_csv("finalmodel16bit.csv")
dataset = dataset.to_numpy()
#normally 628
X = dataset[:,2:152]
y = dataset[:,1]
y=y.astype('int')



qt = QuantileTransformer()
Xnew = qt.fit_transform(X)

model = KNeighborsClassifier(n_neighbors=3, weights='distance')
model.fit(Xnew,y)


#print("Please input IP Address: ")
#ipaddr = input()
#robot = anki_vector.Robot(args.serial)
#print(robot)


#with anki_vector.Robot(args.serial) as robot:


args = anki_vector.util.parse_command_args()

fs = 44100  # Sample rate
seconds = 5  # Duration of recording

directory = '/home/nick/Documents/audio_features/'

with anki_vector.Robot(args.serial) as robot:
#with anki_vector.Robot(serial=s,ip=ipaddr,name=name) as robot:
    while True:

        hue = 0.83
        saturation = 0.76
        robot.behavior.set_eye_color(hue,saturation) #purple
        robot.behavior.set_head_angle(anki_vector.behavior.MAX_HEAD_ANGLE)
        # Capture audio from through microphone
        ##############################################
        audioRecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2, dtype='int16')
        print("Speak!...it's recording")
        sd.wait()  # Wait until recording is finished
        write(directory + 'testlive/snippet.wav', fs, audioRecording)

        print("Waiting 300 milliseconds for audio to be saved and chunked up...")
        time.sleep(.300)

        import os
        import pandas as pd
        from joblib import dump,load
        folder = 'testlive/'
        
        for filename in os.listdir(directory + folder):
            if filename.endswith(".wav") or filename.endswith(".mp3"):
                
                allFeatures = audio_extractors.getFeatures(folder+filename)

                

                if(isinstance(allFeatures,str)):
                    print("------------------------------------------------------\n\n\n\n\n\n\n\nSILENT\n\n\n\n\n\n\n\n------------------------------------------------------")
                    hue = 0.11
                    saturation = 1.00
                    robot.behavior.set_eye_color(hue,saturation) #yellow
                    time.sleep(3.0)
                else:
                    normData = qt.transform(allFeatures.to_numpy()[:,2:152])
                    ourPrediction = model.predict(normData)
                    ourPrediction = int(ourPrediction[0])

                    print(ourPrediction)
                    if(ourPrediction==0):
                        hue = 0.01
                        saturation = 0.95
                        robot.behavior.set_eye_color(hue,saturation) #red
                        time.sleep(3.0)
                        print("------------------------------------------------------\n\n\n\n\n\n\n\nINTERACTIVE\n\n\n\n\n\n\n\n------------------------------------------------------")
                    else:
                        hue = 0.42
                        saturation = 1.00
                        robot.behavior.set_eye_color(hue,saturation) #green
                        time.sleep(3.0)
                        print("------------------------------------------------------\n\n\n\n\n\n\n\nNOT INTERACTIVE\n\n\n\n\n\n\n\n--------------------------------------------------")
