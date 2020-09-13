# Preprocessing dependencies
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report,confusion_matrix
from itertools import product

# Classifiers dependencies 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn import tree
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def logistic_regression(X_Train, X_Test, y_train, y_test):
    trainedmodel = LogisticRegression().fit(X_Train, y_train)
    predictions = trainedmodel.predict(X_Test)
    print(confusion_matrix(y_test,predictions))
    print(classification_report(y_test,predictions))

def random_forest(X_Train, X_Test, y_train, y_test, n_estimators=10, max_depth=5, max_features=1):
    trainedforest = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features).fit(X_Train, y_train)
    predictionforest = trainedforest.predict(X_Test)
    print(confusion_matrix(y_test,predictionforest))
    print(classification_report(y_test,predictionforest))

