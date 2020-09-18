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
        while True:
            print("(THINKING = -1) Set Vector's eye color to BLUE...")
            robot.behavior.set_eye_color(hue=0.55, saturation=0.85)
            robot.behavior.set_head_angle(anki_vector.behavior.MAX_HEAD_ANGLE)

            # capture audio from mic
            sample = listen(seconds, fs, rec_directory, overwrite=True)
            time.sleep(.300)

            # parse sample # DEBUG
            features = extract_file_features(file=sample, target=-1, filter_band = True, filter_directory = rec_directory + 'filtered/') 

            # classify samples
            prediction = -1
            if (features != 'silent'):
                clf = joblib.load('models/Linear SVM.sav')
                prediction = get_classification(features, clf)

            # output classification
            if (prediction == 0): # non-interactive
                print("(NON-INTERACTIVE = 0) Set Vector's eye color to ORANGE...")
                robot.behavior.set_eye_color(hue=0.05, saturation=0.95)
            elif (prediction == 1): # (default) interactive
                print("(INTERACTIVE = 1) Set Vector's eye color to PURPLE...")
                robot.behavior.set_eye_color(hue=0.83, saturation=0.76)
            elif (prediction == -1):
                print("(SILENT) Set Vector's eye color to YELLOW...")
                robot.behavior.set_eye_color(hue=0.11, saturation=0.95)
            
            time.sleep(3.0)

if __name__ == '__main__':
    main()
