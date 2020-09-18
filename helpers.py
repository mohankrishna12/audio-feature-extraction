# file support
import os
from os import listdir
from os.path import isfile, join, basename, splitext
import librosa

##################################################
#              AUXILIARY FUNCTIONS               #
##################################################

def remove_extension(file):
    return os.path.splitext(file)[0]

def path_leaf(path):
    head, tail = os.path.split(path)
    return tail or os.path.basename(head)

def get_files(directory, valid_exts):
    return ([directory + x for x in 
        [f for f in listdir(directory) if (isfile(join(directory, f)) # get files only
        and bool([ele for ele in valid_exts if(ele in f)])) ]         # with valid extension
        ])  

def generate_file_name(outDir, inFile, lowcut, highcut, suffix = "_filtered", extension = ".wav"):
    return outDir + remove_extension(path_leaf(inFile)) + '_' + str(int(lowcut)) + '-' + str(int(highcut)) + suffix + extension