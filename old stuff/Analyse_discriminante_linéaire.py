import pandas as pd
import numpy as β
import csv
β.set_printoptions(precision=4)
from matplotlib import pyplot as traceuse
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MaxNLocator
import seaborn as sns
sns.set()
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

datafile = '/Users/emmanueltoksadeniran/Desktop/Audio_Project/Input_Files/ComboFeatures.csv'
PlotSpot = '/Users/emmanueltoksadeniran/Desktop/Audio_Project/Output_Files/Charts/'
filename = "LDA_Chart"


def plotter(attributes, firstColumn, secondColumn, encodedClass, filename, PlotSpot):
    colorPalette = sns.color_palette("muted", n_colors=15)
    cmap = ListedColormap(sns.color_palette(colorPalette).as_hex())
    colors = β.random.randint(0,15,attributes)
    # traceuse.scatter(CorrelationCoefficient, CorrelationCoefficient, c=colors, cmap=cmap)
    traceuse.scatter(firstColumn, secondColumn, c=encodedClass, cmap=cmap, alpha=0.7, edgecolors='b')
    traceuse.title('Linear Discriminant Analysis')
    traceuse.ylabel('LDA Values')
    traceuse.yticks(fontsize=8, rotation=0)
    traceuse.xlabel('LDA Values')
    traceuse.xticks(fontsize=8, rotation=0)
    name = PlotSpot + str(filename) + ".pdf"
    traceuse.savefig(name, bbox_inches='tight')
    traceuse.show(block=False)
    traceuse.pause(2)
    traceuse.close("all")
    return

dataset = pd.read_csv(datafile)
data = dataset.iloc[:,1:].copy()
target = dataset['categorization'].copy()
target_names = ['Mediated', 'Immediate']

table = pd.DataFrame(data=data, columns=data.columns)

# print("Target: ", "\n", target)
# print("Table: ", "\n", table)
# print("Target Names: ", target_names)

classification = pd.Categorical.from_codes(target, target_names)

taxonomic = table.join(pd.Series(classification, name='taxonomy'))

averagesOfFeatures = pd.DataFrame(columns=target_names)
for c, rows in taxonomic.groupby('taxonomy'):
    averagesOfFeatures[c] = rows.mean()
print(averagesOfFeatures)
print("length of target/colums/features: ", len(table.columns))
attributes = len(table.columns)



within_class_scatter_matrix = β.zeros((len(table.columns),len(table.columns)))
for c, rows in taxonomic.groupby('taxonomy'):
    rows = rows.drop(['taxonomy'], axis=1)
    
    s = β.zeros((len(table.columns),len(table.columns)))

for index, row in rows.iterrows():
    x, mc = row.values.reshape(len(table.columns),1), averagesOfFeatures[c].values.reshape(len(table.columns),1)
    s += (x - mc).dot((x - mc).T)
    
    within_class_scatter_matrix += s

# for col in data.columns: 
#     print(col) 

feature_means = taxonomic.mean()
between_class_scatter_matrix = β.zeros((len(table.columns),len(table.columns)))
for c in averagesOfFeatures:    
    n = len(taxonomic.loc[taxonomic['taxonomy'] == c].index)
    
    mc, m = averagesOfFeatures[c].values.reshape(len(table.columns),1), feature_means.values.reshape(len(table.columns),1)
    
    between_class_scatter_matrix += n * (mc - m).dot((mc - m).T)



eigen_values, eigen_vectors = β.linalg.eig(β.linalg.inv(within_class_scatter_matrix).dot(between_class_scatter_matrix))



pairs = [(β.abs(eigen_values[i]), eigen_vectors[:,i]) for i in range(len(eigen_values))]
pairs = sorted(pairs, key=lambda x: x[0], reverse=True)
for pair in pairs:
    print(pair[0])



eigen_value_sums = sum(eigen_values)
print('Explained Variance')
for i, pair in enumerate(pairs):
    print('Eigenvector {}: {}'.format(i, (pair[0]/eigen_value_sums).real))



w_matrix = β.hstack((pairs[0][1].reshape(len(table.columns),1), pairs[1][1].reshape(len(table.columns),1))).real



X_lda = β.array(table.dot(w_matrix))



le = LabelEncoder()
encodedClass = le.fit_transform(taxonomic['taxonomy'])



firstColumn = X_lda[:,0]
secondColumn = X_lda[:,1]

plotter(attributes, firstColumn, secondColumn, encodedClass, filename, PlotSpot)
