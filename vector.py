# dependencies 
from extractors import *
from parsers import *
from classifiers import *
from listen import *
from helpers import *

import joblib
import time
import anki_vector

# fs = 44100
# seconds = 5
rec_directory = "experiment/recordings/"
samples = get_files('experiment/samples/', ['.wav']) # samples to play and record
results_csv = "experiment/results.csv"

def main():
    print("HELLO")
    args = anki_vector.util.parse_command_args()

    results_df = pd.DataFrame(columns = ['Sample', 'Prediction', 'Confidence', 'Time Spent'])

    with anki_vector.Robot(args.serial) as robot:
        for sample in samples:
            print("PLAYING " + sample)

            print("(THINKING = -1) Set Vector's eye color to BLUE...")
            robot.behavior.set_eye_color(hue=0.55, saturation=0.85)
            robot.behavior.set_head_angle(anki_vector.behavior.MAX_HEAD_ANGLE)
            time.sleep(3.0)

            # playback on new sample
            item = playback(sample, rec_directory)
            
            # capture audio from mic
            # sample = listen(seconds, fs, rec_directory, overwrite=False)
            # time.sleep(3.0)

            time_start = time.time()

            # parse sample # DEBUG
            features = extract_file_features(file=item, target=-1, filter_band = True, filter_directory = rec_directory + 'filtered/') 

            # classify samples
            prediction = -1
            confidence = -1
            if (not(isinstance(features, str))): #(features != 'silent'):
                clf = joblib.load('models/Linear SVM.sav')
                prediction = get_classification(features, clf)
                try: 
                    confidence = clf.predict_proba(features)[prediction]
                except Exception as ex:
                    print("Model selected may not support prediction probabilities", type(ex))

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
            
            time_end = time.time()

            results_df = results_df.append({'Sample' : item, 'Prediction' : prediction, 'Confidence' : confidence, 'Time Spent' : time_end - time_start}, ignore_index=True) 

            time.sleep(3.0)
        
        results_df.to_csv(results_csv)

if __name__ == '__main__':
    main()
