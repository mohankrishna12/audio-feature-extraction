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
# librosa
import librosa
import librosa.display
import audio_extractors

'''
def create_mel_spectrogram(audio, hop_length, n_fft, sampling_rate, n_mels):
    mel = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    S = librosa.feature.melspectrogram(audio, sr=sampling_rate, n_fft=n_fft,
                                       hop_length=hop_length,
                                       n_mels=n_mels)
    S_DB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length,
                             x_axis='time', y_axis='mel')
'''


fs = 44100  # Sample rate
seconds = 5  # Duration of recording
#hop_length = 512
#nfft = 2048
#n_mels = 128
fig, ax = plt.subplots(1)
fig.subplots_adjust(left=0, right=10, bottom=0, top=10)
ax.axis('tight')
ax.axis('off')
directory = '/home/nick/Documents/audio_features/'

# Capture audio from through microphone
##############################################
audioRecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
print("Speak!...it's recording")
sd.wait()  # Wait until recording is finished
write(directory + 'LongAudio/longOutputAudio.wav', fs, audioRecording)

print("Waiting 300 milliseconds for audio to be saved and chunked up...")
time.sleep(.300)
# print("Waiting 2 seconds while snippets are chunked up...")
# time.sleep(3)
##############################################


# Creating audio snippets
##############################################
from pydub import AudioSegment
from pydub.utils import make_chunks

myaudio = AudioSegment.from_file(directory + 'LongAudio/longOutputAudio.wav', "wav")
samplerate = 44100
myaudio = myaudio.set_frame_rate(samplerate)
# myaudio, sr = librosa.load(filename, sr=8000)
chunk_length_ms = 5000
chunks = make_chunks(myaudio, chunk_length_ms)

for i, chunk in enumerate(chunks):
    chunk_name = 'snippet' + "{0}.wav".format(i)
    print("Exporting...", chunk_name)
    chunk.export(directory + 'AudioSnippets/' + chunk_name, format="wav")
##############################################

#allFeatures = audio_extractors.getFeatures("AudioSnippets/snippet0.wav")
#print(allFeatures.shape)


# Create and save features spreadsheets
##############################################

import os
import pandas as pd
from joblib import dump,load
for filename in os.listdir(directory + 'AudioSnippets/'):
    if filename.endswith(".wav") or filename.endswith(".mp3"):
        #snippet, sr = librosa.load(directory + 'AudioSnippets/' + filename, sr=44100)
        # Call Feature Extraction Functions here
        #dataframe = pd.DataFrame() # pass return of function to dataframe.
        #filename = os.path.splitext(filename)[0]
        #print("Filename extension lopped off...", filename)
        #filename = '{0}.csv'.format(filename)
        #featuresFile = os.path.join(directory + 'Spectrograms/', filename)
        #dataframe.to_csv(featuresFile)
        # np.savetxt(featuresFile, delimiter=",")
        # np.savetxt(featuresFile, delimiter=",", fmt="%s")
        allFeatures = audio_extractors.getFeatures('AudioSnippets/'+filename)
        print(allFeatures.shape)
        clf = load('ourSVM.joblib')

        ourPrediction = clf.predict(allFeatures)
        print("-----------------------------------------------------")
        print("The prediction for this image is: ", ourPrediction[0])
        print("The label is 1 for interactive | 0 for not")


        continue
    else:
        continue
'''
##############################################

# print("Waiting 3 seconds while features spreadsheets are created...")
# time.sleep(3)

# Predict from live audio
##############################################
import pickle


# load model
with open('model.pkl', 'rb') as f:
    pickleloaded = pickle.load(f)

for featuresFile in os.listdir(directory + 'Features/'):
    if featuresFile.endswith(".csv") or featuresFile.endswith(".txt"):
        dataset = pd.read_csv(featuresFile)
        dataset = dataset.drop([dataset.columns[0], dataset.columns[1]], axis=1)
        picklePrediction = pickleloaded.predict(featuresFile)
        print("The prediction for this image is: ", picklePrediction)
        print("The label is 1 for interactive | 0 for not")

# featuresFolder = os.listdir(directory + 'Features/')
# for i, featuresFile in enumerate(featuresFolder):
#     if  featuresFile.split('.')[1] == 'csv':
#         dataset = pd.read_csv(featuresFile)
#         dataset = dataset.drop([dataset.columns[0], dataset.columns[1]], axis=1)
#         picklePrediction = pickleloaded.predict(featuresFile)
#         print("The prediction for this image is: ", picklePrediction)
#         print("The label is 1 for interactive | 0 for not")

##############################################

# Replay audio that was recorded and passed for prediction
##############################################

# import sounddevice as sd
import soundfile as sf

# filename = 'testOutputAudio.wav'
# filename = 'DrFerburRevisitsHisCryingBabyTheory.wav'
# Extract data and sampling rate from file
data, fs = sf.read(directory + 'LongAudio/longOutputAudio.wav', dtype='float32')
sd.play(data, fs)
status = sd.wait()  # Wait until file is done playing
'''