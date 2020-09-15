# Dependencies 
from extractors import *
from parsers import *
from classifiers import *
from listen import *
from helpers import *

# Parsing
df_audio = read_features_from_file("all_features.csv")

# (Examples)
# get overall features with mono-tize preprocessing step
# electronic_features_mono = pd.concat(extract_dir_overall_features("audio/Electronic/", ['.mp3', '.wav', '.flac'], 0, False, True, True, 'audio/tests/Electronic/'))
# organic_features_mono = pd.concat(extract_dir_overall_features("audio/Organic/", ['.mp3', '.wav', '.flac'], 1, False, True, True, 'audio/tests/Organic/'))
# chime_features = pd.concat(extract_dir_overall_features("chime/clips/", ['.wav'], 1, False, False, True, 'chime/clips/tests/'))
# df_audio = pd.concat([electronic_features_mono, organic_features_mono], axis = 0)

# get overall features on all files (mono + stereo)
# o_electronic_features_stereo = extract_dir_overall_features("audio/Electronic/", ['.mp3', '.wav', '.flac'], 0)
# o_organic_features = extract_dir_overall_features("audio/Organic/", ['.mp3', '.wav', '.flac'], 1)

# Preprocessing
from sklearn.preprocessing import LabelEncoder, QuantileTransformer
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve, f1_score, auc
from itertools import product
from sklearn import model_selection

# Naming Classifiers
names = [
    'K Nearest Neighbors', 
    #'Linear SVM', 
    #'RBF SVM',
    #'Gaussian Process', 
    #'Decision Tree', 
    #'Random Forest', 
    #'Neural Net', 
    #'AdaBoost', 
    #'Naive Bayes', 
    #'QDA'
]

# Defining Classifier Parameters
classifiers = [
    KNeighborsClassifier(3), # number of neighbors = 3
    #SVC(kernel='linear', C=0.025, probability=True), # linear kernel with regularization/misclassification error = 0.025
    #SVC(gamma=2, C=0.025, probability=True), # looser SVM with higher regularization
    #GaussianProcessClassifier(1.0 * RBF(1.0)), # RBF kernel
    #DecisionTreeClassifier(max_depth=5),
    #RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1), # estimators = # of trees in the forest, max_features = # of features to consider when looking for best split
    #MLPClassifier(alpha=0.025, max_iter=1000), # multilayer perceptron with L2 penalty/regularization = 1, max_iter = limit as solver iterates until convergence
    #AdaBoostClassifier(), 
    #GaussianNB(),
    #QuadraticDiscriminantAnalysis()
]

tests = [
    { "name": "Other Test", "file": "datasets/morechimeandtv.csv", "results": [] },
    { "name": "Podcast Test", "file": "datasets/drama_podcast_reality.csv", "results": [] }
]

def evaluate_classifiers(names, classifiers, dataset, tests, results):
    models = []

    # Dataframe components
    scores = []
    accuracy_vals = []
    recall_vals = []
    specificity_vals = []
    precision_vals = []
    false_positive_rates = []
    false_negative_rates = []
    cv_means = []
    cv_stds = []
    no_skill_aucs = []
    logistic_aucs = []
    prc_aucs = []
    prc_f1_scores = []
    lr_precisions = []
    lr_recalls = []

    # Split data
    X = dataset.drop(['target'], axis = 1).values
    y = dataset['target']
    X = QuantileTransformer(output_distribution='normal').fit_transform(X)
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.5, random_state=42)
    print("completing dataset test/train split")

    for name, clf in zip(names, classifiers):
        
        # train classifier
        clf.fit(X_train, y_train)
        
        score = clf.score(X_test, y_test)
        scores.append(score)
        
        # build classifier model
        y_hat = clf.predict(X_test)
        models.append(clf) # save model
        
        # compute basic performance metrics
        tn, fp, fn, tp = confusion_matrix(y_test, y_hat).ravel()
        accuracy_vals.append((tn + tp) / (tn + fp + fn + tp))       # accuracy
        recall_vals.append(tp / (fn + tp))                          # recall
        specificity_vals.append(tn / (tn + fp))                     # specificity
        precision_vals.append(tp / (fp + tp))                       # precision
        false_positive_rates.append(fp / (tn + fp))                 # false positive rate
        false_negative_rates.append(fn / (fn + tp))                 # false negative rate
        print(name, classification_report(y_test, y_hat))
        
        # cross validation score
        kfold = model_selection.KFold(n_splits=10)                  
        cv_score = model_selection.cross_val_score(clf, X, y, cv=kfold)
        cv_means.append(cv_score.mean() * 100.0)
        cv_stds.append(cv_score.std() * 100.0)
        
        # logistical roc auc
        lr_probs_roc = clf.predict_proba(X_test)[:, 1]              
        lr_auc_roc = roc_auc_score(y_test, lr_probs_roc)
        logistic_aucs.append(lr_auc_roc)

        # no skill roc auc  
        ns_probs_roc = [0 for _ in range(len(y_test))]                  
        ns_auc_roc = roc_auc_score(y_test, ns_probs_roc)
        no_skill_aucs.append(ns_auc_roc)

        # roc auc curves
        # ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)           
        # lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
        
        # precision-recall curves
        lr_precision, lr_recall, _ = precision_recall_curve(y_test, lr_probs_roc)
        lr_precisions.append(lr_precision)
        lr_recalls.append(lr_recall)
        lr_f1_prc, lr_auc_prc = f1_score(y_test, y_hat), auc(lr_recall, lr_precision)
        prc_aucs.append(lr_auc_prc)
        prc_f1_scores.append(lr_f1_prc)
        no_skill = len(y_test[y_test==1]) / len(y_test)

        # testing suite
        for test in tests:
            samples = pd.read_csv(test["file"])
            samples_X = samples.drop(['target', 'id'], axis = 1).values
            samples_y = samples['target']
            samples_X = QuantileTransformer(output_distribution='normal').fit_transform(samples_X)
            predictions = clf.predict(samples_X)
            print("TEST (", test["name"], ")", classification_report(samples_y, predictions))
            tn, fp, fn, tp = confusion_matrix(samples_y, predictions).ravel()
            test["results"].append((tn + tp) / (tn + fp + fn + tp))
        
    print("compiling performance dataframe")
    df_performance = pd.DataFrame({
        'classifier': names, 
        'score': scores, 
        'accuracy': accuracy_vals, 
        'recall': recall_vals,
        'specificity': specificity_vals,
        'precision': precision_vals,
        'FPR': false_positive_rates,
        'FNR': false_negative_rates, 
        'CV-10 mean': cv_means,
        'CV-10 std': cv_stds, 
        'No Skill ROC AUC': no_skill_aucs,
        'Logistic ROC AUC': logistic_aucs,
        'Precision-Recall AUC': prc_aucs,
        'F1': prc_f1_scores
    })

    for test in tests:
        df_performance[test["name"]] = test["results"]

    print("exporting performance dataframe")
    df_performance.to_csv(results)    

    return models

# get predicted class of newly recorded sample
def get_classification(features, clf):
    sample_X = features.drop(['target', 'id'], axis = 1).values
    #sample_X = QuantileTransformer(output_distribution='normal').fit_transform(sample_X)
    prediction = clf.predict(sample_X)
    return prediction

# save models
models = evaluate_classifiers(names, classifiers, df_audio, tests, "performance_metrics.csv")

# play and record environmental samples
fs = 44100  # sample rate
audio_samples = get_files('tests/', ['.wav']) # files to replay
write_directory = 'tests/recordings/' # dir to write to

# play and record directory of samples --> TODO:( parse --> classify )
# record_directory(items, write_directory, fs)

# play --> record --> parse --> classify each sample
# for audio_sample in audio_samples:

    # get environmental sample
#    loc = record_file(audio_sample, write_directory, fs)

    # extract features from environmental sample
#    features = extract_file_features(file=loc, target=-1) #, filter_band = True, filter_directory = 'tests/filters/')
#    features.to_csv("tests/features/" + os.path.splitext(path_leaf(loc))[0] + ".csv")

classifications = pd.read_csv("tests/classifications.csv")
features_directory = "tests/features/"
new_samples = get_files(directory = features_directory, valid_exts = ['.csv'])
for new_sample in new_samples:
    features = pd.read_csv(new_sample)
    for model in models:
        classification = get_classification(features, model)
        print("PATH LEAF RETURNS (SAMPLE): ", os.path.splitext(path_leaf(new_sample))[0])
        print("PATH LEAF RETURNS (CLASSI): ", os.path.splitext(path_leaf(classifications['file'][0]))[0])
        print("TRUE TARGET: ", (classifications.loc[classifications['file'] == (os.path.splitext(path_leaf(new_sample))[0] + ".wav")]["target"])[0])
        print("EST. CLASSI: ", classification[0])

print("DONE.")