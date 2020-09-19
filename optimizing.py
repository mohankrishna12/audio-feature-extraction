# dependencies
import pandas as pd 
import numpy as np 

# preprocessing
from sklearn.preprocessing import QuantileTransformer

# for univariate selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler

# for extra trees selection
from sklearn.ensemble import ExtraTreesClassifier

# for recursive feature elimination
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# for LASSO
from sklearn.feature_selection import SelectFromModel

# for KNN
from sklearn.neighbors import KNeighborsClassifier

# for random forest selection
from sklearn.ensemble import RandomForestClassifier

# for selector evaluation
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# for grid search / evaluation
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import f_classif
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from parsers import *

##################################################
#                   SELECTORS                    #
##################################################

def univariate_selector(X, y, k):
    chi_selector = SelectKBest(score_func=chi2, k=k)
    
    # process feature set
    X_norm = MinMaxScaler().fit_transform(X)
    
    # apply chi class
    fit_norm = chi_selector.fit(X_norm,y)
    chi_support = chi_selector.get_support()

    importances = pd.Series(fit_norm.scores_, index=X.columns)
    return importances.nlargest(k) 

# mutual information feature selection
def mutualinfo_selector(X, y, k):
    f_selector = SelectKBest(score_func=mutual_info_classif, k=k)
    
    # process feature set
    X_norm = MinMaxScaler().fit_transform(X)

    # apply selector class
    fit_norm = f_selector.fit(X_norm,y)
    f_support = f_selector.get_support()

    importances = pd.Series(fit_norm.scores_, index=X.columns)
    return importances.nlargest(k)

def extratrees_selector(X, y, k):
    model = ExtraTreesClassifier()
    model.fit(X, y)
    importances = pd.Series(model.feature_importances_, index=X.columns)
    return importances.nlargest(k)

# use if suspecting parametric/linear correlation
def pearson_selector(X, y, k):
    pearson_corrs = []
    feature_names = X.columns.tolist()

    # calculate the correlation of each feature with target
    for f in feature_names:
        pearson_corr = np.corrcoef(X[f], y)[0, 1]
        pearson_corrs.append(pearson_corr)

    # replace NaN with 0
    pearson_corrs = [0 if np.isnan(i) else i for i in pearson_corrs]

    # feature name
    selected_features = X.iloc[:,np.argsort(np.abs(pearson_corrs))].columns.tolist()

    # feature support
    cor_support = [True if i in selected_features else False for i in feature_names]

    # feature selection
    importances = pd.Series(pearson_corrs, index=X.columns)
    # allFeatureScores = pd.DataFrame({'feature': X.columns, 'score': pearson_corrs})
    # featureScores = allFeatureScores.loc[allFeatureScores.feature.isin(selected_features)]

    return importances.nlargest(k)

def recursiveelim_selector(X, y, k, step=10, verbose=0):
    rfe_selector = RFE(estimator=LogisticRegression(), n_features_to_select=k, step=step, verbose=verbose)
    rfe_selector.fit(X, y)
    rfe_support = rfe_selector.get_support()
    rfe_feature = X.loc[:,rfe_support].columns.tolist()
    importances = pd.Series(rfe_selector.estimator_.coef_[0], index=rfe_feature)
    return importances.nlargest(k)

def lasso_selector(X, y, k):
    lr_selector = SelectFromModel(estimator=LogisticRegression(penalty='l2', solver='newton-cg'), max_features=k)
    lr_selector.fit(X, y)
    lr_support = lr_selector.get_support()
    lr_feature = X.loc[:,lr_support].columns.tolist()
    importances = pd.Series(lr_selector.estimator_.coef_[0], index=X.columns)
    return importances.nlargest(k)

def randomforest_selector(X, y, k):
    rf_selector = SelectFromModel(RandomForestClassifier(n_estimators=100), max_features=10)
    rf_selector.fit(X, y)
    rf_support = rf_selector.get_support()
    rf_feature = X.loc[:,rf_support].columns.tolist()

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)

    importances = pd.Series(model.feature_importances_, index=X.columns)
    return importances.nlargest(k)

# feature selection
def select_features(X_train, y_train, X_test, k, fs):
	# learn relationship from training data
    fs.fit(X_train, y_train)
	# transform train input data
    X_train_fs = fs.transform(X_train)
	# transform test input data
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs

def evaluate_selector(X, y, k, model, selector):
    X = QuantileTransformer(output_distribution='normal').fit_transform(X)
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.5, random_state=42)
    X_train_fs, X_test_fs, fs = \
        select_features(X_train, y_train, X_test, k, selector)
    model.fit(X_train_fs, y_train)
    yhat_fs = model.predict(X_test_fs)
    return X_train_fs, X_test_fs, fs, accuracy_score(y_test, yhat_fs)

# evaluate a give model using cross-validation
def evaluate_model(model, X, y):
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
	return scores

# get results of dimensionality results 
def grid_search_distribution(X, y, k, model, fs, model_name, fs_name, output_loc):
    # define the number of features to evaluate
    num_features = [i+1 for i in range(X.shape[1])]

    # enumerate each number of features
    results = list()
    grid_df = pd.DataFrame(columns = ['Features', 'CV-Mean', 'CV-STD'])

    for k in num_features:
        # create pipeline
        # model = LogisticRegression(solver='liblinear')
        # fs = SelectKBest(score_func=f_classif, k=k)
        pipeline = Pipeline(steps=[(fs_name,fs), (model_name, model)])
        
        # evaluate the model
        scores = evaluate_model(pipeline, X, y)
        results.append(scores)
        print('>%d features --> %.3f (%.3f)' % (k, np.mean(scores), np.std(scores)))
        grid_df = grid_df.append({'Features' : k, 'CV-Mean' : np.mean(scores), 'CV-STD' : np.std(scores)}, ignore_index=True) 
        
    grid_df.to_csv(output_loc)
    return grid_df

def grid_search(X, y, model, fs, model_name, fs_name):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

    # define the pipeline to evaluate
    # model = LogisticRegression(solver='liblinear')  # classifier
    # fs = SelectKBest(score_func=f_classif)          # feature selector
    pipeline = Pipeline(steps=[(fs_name,fs), (model_name, model)])

    # define the grid
    grid = dict()
    grid[fs_name + '__k'] = [i+1 for i in range(10)] #range(X.shape[1])]

    # define the grid search
    search = GridSearchCV(pipeline, grid, scoring='accuracy', n_jobs=-1, cv=cv)

    # perform the search
    results = search.fit(X, y)

    # summarize best
    return results

##################################################
#                    TESTING                     #
##################################################
'''
dataset = read_features_from_file("all_features.csv")

X = dataset.drop(['target'], axis = 1)
y = dataset['target']

model = KNeighborsClassifier(3)
selector = SelectKBest(score_func=f_classif)
results = grid_search(X, y, model, selector, "KNN", "ANOVA")

print('Best Mean Accuracy: %.3f' % results.best_score_)
print('Best Config: %s' % results.best_params_)
print('Best No. of Dimensions: %d' % np.array(list(results.best_params_.values()))[0])

k = np.array(list(results.best_params_.values()))[0] # number of features to select
selector = SelectKBest(score_func=f_classif, k=k)

X_train_fs, X_test_fs, fs, accuracy = evaluate_selector(X, y, k, model, selector)

# record scores for the features
print('compiling feature distribution output')
k_df = pd.DataFrame(columns = ['Feature', 'K', 'p'])
for i in range(len(fs.scores_)):
    k_df = k_df.append({'Feature' : X.columns[i], 'K' : fs.scores_[i], 'p' : fs.pvalues_[i]}, ignore_index=True)    
k_df.to_csv('features_scores.csv')

print('printing k important features')
importances = k_df.nlargest(k, 'K')
print(importances)
'''
# using selection methods
# print(univariate_selector(X, y, 5))     # univariate selection
# print(extratrees_selector(X, y, 5))     # extra trees classifier selection
# print(pearson_selector(X, y, 5))        # pearson correlation selection
# print(recursiveelim_selector(X, y, 5))  # recursive elimination selection
# print(lasso_selector(X, y, 5))          # LASSO selector 
# print(randomforest_selector(X, y, 5))   # random forest classifier selector
# print(mutualinfo_selector(X, y, 5))       # mutual information gain selector