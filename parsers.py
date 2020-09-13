# essentia
# import essentia
# from essentia.standard import *
# import essentia.standard as es

# librosa
import librosa
import librosa.display

# file support
import os
from os import listdir
from os.path import isfile, join, basename, splitext

# analysis
import IPython
import numpy as np
import pandas as pd
import scipy
import sklearn

# plots
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("ticks")
sns.set_palette("bright")

##################################################
#            PARSING AUDIO DIRECTORY             #
##################################################

def extract_dir_overall_features(
    directory,   # parent directory of main audio files
    extensions,  # filter directory by file extensions
    target,      # directory represents which class
    mono_read = False, mono_parse = False, # mono-tize files
    filter_band = False, filter_directory = '', filter_lowcut = 20.0, filter_highcut = 250.0): 
    
    all_features = []
    files = []
    if mono_read: # get only files with mono in name
        files = [i for i in get_files(directory, extensions) if "_mono" in i]
    else: # get all files without mono
        files = [i for i in get_files(directory, extensions) if not "_mono" in i]
    
    # get EBM values
    # EBM_values = pd.DataFrame()
    # if (target == 0):
    #     EBM_values = pd.read_csv("electronicEBMs.csv", delimiter=',', index_col=0, names = ['filename', 'ebmVal'])
    # else: EBM_values = pd.read_csv("organicEBMs.csv", delimiter=',', index_col=0, names = ['filename', 'ebmVal'])

    print('\nReading directory', directory)
    for file in files:
        print('\tParsing features from', file)
        try:
            # determine target and identifier
            print('\t\t generating row identifiers')
            description = pd.DataFrame({
                'id':[file.split('/')[-1].split('.')[0]],
                'target':[target]
            }, index=[0])

            # preprocessing here
            curr_file = file
            if mono_parse: # will parse and create mono file
                print('\t\t parsing file for mono conversion')
                y, _ = librosa.load(file, sr=44100.)
                mono_stream = librosa.core.to_mono(y)
                mono_filename = directory + remove_extension(path_leaf(file)) + "_mono.wav"
                print('\t\t saving file for mono file as', mono_filename)
                sf.write(mono_filename, mono_stream, 44100)
                curr_file = mono_filename
                print('\t\t finished mono parsing')
            
            # extract sub-band features
            filtered_features = pd.DataFrame()
            if filter_band:
                print('\t\t extracting frequency band from', curr_file)
                loc = generate_file_name(filter_directory, file, filter_lowcut, filter_highcut)
                extract_freq_band(file, filter_lowcut, filter_highcut, loc)
                print('\t\t saving frequency band as', loc)
                filtered_features = pd.concat(
                    (extract_librosa_features(loc), 
                     #extract_essentia_features(loc), 
                     extract_hum_features(loc, [[.1, .55], [.1, .25], [.1, .75]]),
                     extract_discontinuity_features(loc), 
                     extract_clicks_features(loc),
                    ), axis = 1).add_prefix('band_')
                print('\t\t finished extracting features from frequency band')
            
            # extract overall spectra features
            overall_features = pd.concat(
                (description,
                 extract_librosa_features(curr_file), 
                 #extract_essentia_features(curr_file), 
                 extract_hum_features(curr_file, [[.1, .55], [.1, .25], [.1, .75]]),
                 extract_discontinuity_features(curr_file), 
                 extract_clicks_features(curr_file),
                 extract_ebm_features(EBM_values.loc[os.path.basename(curr_file), "ebmVal"]),
                 filtered_features
                ), axis = 1)
            print('\t\t finished extracting features from overall spectrum')
            
            # add to feature list
            all_features.append(overall_features)
        except Exception as e:
            print(e)
            
    return all_features

##################################################
#           READ EXISTING FEATURE CSV(S)         #
##################################################
def read_features_from_file(csv_path):
    return pd.read_csv(csv_path, index_col = 0)

def read_features_from_files(csv_paths):
    for csv_path in csv_paths:
        li = []
        df = pd.read_csv(csv_path, index_col=None, header=0)
        li.append(df)
    frame = pd.concat(li, axis=0, ignore_index=True)
    return frame
