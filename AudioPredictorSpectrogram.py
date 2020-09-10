print("Aloha")
import sounddevice as sd
from scipy.io.wavfile import write
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import time

def create_mel_spectrogram(audio, hop_length, n_fft, sampling_rate, n_mels):
    mel = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    S = librosa.feature.melspectrogram(audio, sr=sampling_rate, n_fft=n_fft,
                                       hop_length=hop_length,
                                       n_mels=n_mels)
    S_DB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length,
                             x_axis='time', y_axis='mel')

fs = 44100  # Sample rate
seconds = 15  # Duration of recording
hop_length = 512
nfft = 2048
n_mels = 128
fig, ax = plt.subplots(1)
fig.subplots_adjust(left=0, right=10, bottom=0, top=10)
ax.axis('tight')
ax.axis('off')
directory = '/Users/emmanueltoksadeniran/Desktop/BinaryClassication/LivePlayground/'

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


# Create and save spectrograms
##############################################
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

for filename in os.listdir(directory + 'AudioSnippets/'):
    if filename.endswith(".wav") or filename.endswith(".mp3"):
        snippet, sr = librosa.load(directory + 'AudioSnippets/' + filename, sr=44100)
        create_mel_spectrogram(snippet, hop_length, nfft, sr, n_mels)
        print("Spectrogram for", filename + " file created")
        print("Filename at iteration b4 lopping off...", filename)
        filename = os.path.splitext(filename)[
            0]  # Test before next run on a few files to see if it lops off the file extension
        print("Filename extension lopped off...", filename)
        filename = '{0}.png'.format(filename)
        print("Filename with PNG...", filename)
        image = os.path.join(directory + 'Spectrograms/', filename)
        fig.savefig(image, bbox_inches='tight', transparent=False, pad_inches=0.0)
        continue
    else:
        continue
##############################################

# print("Waiting 3 seconds while spectrograms are created...")
# time.sleep(3)

# Predict from live audio
##############################################
from keras.models import load_model
import cv2
from PIL import Image
SIZE = 150
# # load model
model = load_model('interactivity_model_300epochs.h5')

interactivity_images = os.listdir(directory + 'Spectrograms/')
for i, image_name in enumerate(interactivity_images):    #Remember enumerate method adds a counter and returns the enumerate object
    if (image_name.split('.')[1] == 'png'):
        image = cv2.imread(directory + 'Spectrograms/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((SIZE, SIZE))
        input_img = np.expand_dims(image, axis=0)  # Expand dims so the input is (num images, x, y, c)
        print("The prediction for this image is: ", model.predict(input_img))
        print("The label is 0 for interactive | 1 for not")

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
