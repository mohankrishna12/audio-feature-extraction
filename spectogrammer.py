import librosa
import librosa.display
from pydub import AudioSegment
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import argparse

interactiveDirectory = '/Users/emmanueltoksadeniran/Desktop/BinaryClassication/audioFiles/ChoppedInteractive/'
noninteractiveDirectory = '/Users/emmanueltoksadeniran/Desktop/BinaryClassication/audioFiles/ChoppedNoninteractive/'
interactivePath = '/Users/emmanueltoksadeniran/Desktop/BinaryClassication/interactive/'
noninteractivePath = '/Users/emmanueltoksadeniran/Desktop/BinaryClassication/noninteractive/'
# audio_fpath = "../input/audio/audio/"


def create_log_spectrogram(audio, hop_length, n_fft, sampling_rate):
	D = np.abs(librosa.stft(audio, n_fft=n_fft,  hop_length=hop_length))
	DB = librosa.amplitude_to_db(D, ref=np.max)
	librosa.display.specshow(DB, sr=sampling_rate, hop_length=hop_length, 
                         	x_axis='time', y_axis='log')


def create_mel_spectrogram(audio, hop_length, n_fft, sampling_rate, n_mels):
	mel = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
	S = librosa.feature.melspectrogram(audio, sr=sampling_rate, n_fft=n_fft, 
                                   		hop_length=hop_length, 
                                   		n_mels=n_mels)
	S_DB = librosa.power_to_db(S, ref=np.max)
	librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length, 
                         	x_axis='time', y_axis='mel')

def spectrogram(samples, sample_rate, stride_ms = 10.0, 
                          window_ms = 20.0, max_freq = None, eps = 1e-14):

    stride_size = int(0.001 * sample_rate * stride_ms)
    window_size = int(0.001 * sample_rate * window_ms)

    # Extract strided windows
    truncate_size = (len(samples) - window_size) % stride_size
    samples = samples[:len(samples) - truncate_size]
    nshape = (window_size, (len(samples) - window_size) // stride_size + 1)
    nstrides = (samples.strides[0], samples.strides[0] * stride_size)
    windows = np.lib.stride_tricks.as_strided(samples, 
                                          shape = nshape, strides = nstrides)
    
    assert np.all(windows[:, 1] == samples[stride_size:(stride_size + window_size)])

    # Window weighting, squared Fast Fourier Transform (fft), scaling
    weighting = np.hanning(window_size)[:, None]
    
    fft = np.fft.rfft(windows * weighting, axis=0)
    fft = np.absolute(fft)
    fft = fft**2
    
    scale = np.sum(weighting**2) * sample_rate
    fft[1:-1, :] *= (2.0 / scale)
    fft[(0, -1), :] /= scale
    
    # Prepare fft frequency list
    freqs = float(sample_rate) / window_size * np.arange(fft.shape[0])
    
    # Compute spectrogram feature
    ind = np.where(freqs <= max_freq)[0][-1] + 1
    specgram = np.log(fft[:ind, :] + eps)
    return specgram

hop_length = 512
nfft=2048
n_mels = 128

fig,ax = plt.subplots(1)
fig.subplots_adjust(left=0,right=10,bottom=0,top=10)
ax.axis('tight')
ax.axis('off')


for filename in os.listdir(interactiveDirectory):
    if filename.endswith(".wav") or filename.endswith(".mp3"):
        # print(os.path.join(directory, filename))
        print(filename)
        snippet, sr = librosa.load(interactiveDirectory+filename, sr=44100)
        # snippet, sr = librosa.load(filename, sr=44100)
        print("Snippet", snippet)
        # print(type(snippet), type(sr))
        # print(snippet.shape, sr)

        # # create_log_spectrogram(snippet, hop_length, nfft, sr)
        create_mel_spectrogram(snippet, hop_length, nfft, sr, n_mels)
        # spectrogram(snippet, sr, stride_ms = 10.0, window_ms = 20.0, max_freq = None, eps = 1e-14)
        print("Spectrogram for", filename + " file created")

        # # sound = AudioSegment.from_wav(directory+filename)
        # # print("Soundfile loaded", sound)
        filename = 'interactive'+'_'+'{0}.png'.format(filename)
        image = os.path.join(interactivePath, filename)
        print(image)
        fig.savefig(image, bbox_inches='tight', transparent=False, pad_inches=0.0)
        continue
    else:
        continue


for filename in os.listdir(noninteractiveDirectory):
    if filename.endswith(".wav") or filename.endswith(".mp3"):
        # print(os.path.join(directory, filename))
        print(filename)
        snippet, sr = librosa.load(noninteractiveDirectory+filename, sr=44100)
        # snippet, sr = librosa.load(filename, sr=44100)
        print("Snippet", snippet)
        # print(type(snippet), type(sr))
        # print(snippet.shape, sr)

        # # create_log_spectrogram(snippet, hop_length, nfft, sr)
        create_mel_spectrogram(snippet, hop_length, nfft, sr, n_mels)
        # spectrogram(snippet, sr, stride_ms = 10.0, window_ms = 20.0, max_freq = None, eps = 1e-14)
        print("Spectrogram for", filename + " file created")

        # # sound = AudioSegment.from_wav(directory+filename)
        # # print("Soundfile loaded", sound)
        filename = 'noninteractive'+'_'+'{0}.png'.format(filename)
        image = os.path.join(noninteractivePath, filename)
        print(image)
        fig.savefig(image, bbox_inches='tight', transparent=False, pad_inches=0.0)
        continue
    else:
        continue