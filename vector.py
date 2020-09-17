# dependencies 
from extractors import *
from parsers import *
from classifiers import *
from listen import *
from helpers import *

import joblib
import time
import anki_vector

fs = 44100
seconds = 5
rec_directory = "recordings/"

def main():
    args = anki_vector.util.parse_command_args()

    with anki_vector.Robot(args.serial) as robot:
        
        # capture audio from mic
        sample = listen(seconds, fs, rec_directory, overwrite=True)
        time.sleep(.300)

        # parse sample # DEBUG
        # features = extract_file_features(file=sample, target=-1) 
        # features = extract_file_features(file="live tests/3-15.wav", target=-1) # DEBUG: update to full parse
        features = pd.read_csv("live tests/debug/3-15.csv")

        # classify samples
        clf = joblib.load('models/Linear SVM.sav')
        prediction = get_classification(features, clf)

        # output classification
        if (prediction == 0): # non-interactive
            print("Set Vector's eye color to orange...")
            robot.behavior.set_eye_color(hue=0.05, saturation=0.95)
            time.sleep(3.0)
        else: # (default) interactive
            print("Set Vector's eye color to purple...")
            robot.behavior.set_eye_color(hue=0.83, saturation=0.76)
            time.sleep(3.0)

if __name__ == '__main__':
    main()
