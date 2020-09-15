import pandas as pd
import numpy as β
import csv
from matplotlib import pyplot as traceuse
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_iris
from factor_analyzer import FactorAnalyzer
from factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo
import sys, os

# datafile = '/Users/emmanueltoksadeniran/Downloads/bfi.csv'
datafile = '/Users/emmanueltoksadeniran/Desktop/Audio_Project/Input_Files/librosa_features_natural.csv'
# artificial = '/Users/emmanueltoksadeniran/Desktop/Audio_Project/Input_Files/librosa_features_electronic.csv'
# datafile = '/Users/emmanueltoksadeniran/Desktop/Audio_Project/Input_Files/ComboFeatures.csv'
PlotSpot = '/Users/emmanueltoksadeniran/Desktop/Audio_Project/Output_Files/Charts/'


def plotter(numberOfFeatures, Original_Eigenvalues, filename, PlotSpot):
    colorPalette = sns.color_palette("muted", n_colors=25)
    cmap = ListedColormap(sns.color_palette(colorPalette).as_hex())
    colors = β.random.randint(0,15,numberOfFeatures)
    traceuse.scatter(range(1,numberOfFeatures+1),Original_Eigenvalues, c=colors, cmap=cmap)
    traceuse.plot(range(1,numberOfFeatures+1),Original_Eigenvalues)
    traceuse.title('Scree Plot of Audio Features')
    traceuse.xlabel("Factors' Ordinal Numbers")
    # traceuse.xticks(fontsize=8, rotation=0)
    traceuse.ylabel('Eigenvalues')
    # traceuse.yticks(fontsize=8, rotation=0)
    traceuse.grid()
    name = PlotSpot + str(filename) + ".pdf"
    traceuse.savefig(name, bbox_inches='tight')
    traceuse.show(block=False)
    traceuse.pause(5)
    traceuse.close("all")
    return

dataset = pd.read_csv(datafile)
table = pd.DataFrame(data=dataset, columns=dataset.columns)
transposedColumnTitles = table.columns.transpose()
# table['Features'] = transposedColumnTitles
print("Tables Column Length: ", transposedColumnTitles)
filename = os.path.basename(datafile)
names = (str(filename))

dataset.dropna(inplace=True)
# print(dataset.columns, "\n")
# dataset.info()
# print(dataset.head())

# Adequacy Tests:
chi_square_value,p_value=calculate_bartlett_sphericity(dataset)
print("\nThe file named:", names, "has a Bartlett’s Test of Sphericity (p-value close to 0 Desired) of: ", chi_square_value, "and a p-value of: ", p_value, "\n")

kmo_all,kmo_model=calculate_kmo(dataset)
print("The file named:", names, "has a Kaiser-Meyer-Olkin (KMO) Test Score of (>0.6 Desired): ", kmo_model)
numberOfFactors = 5
factorAnalyzed = FactorAnalyzer(n_factors=numberOfFactors, rotation="varimax")
fitData = factorAnalyzed.fit(dataset)
# print("Fit Data: ", fitData)
Original_Eigenvalues, common_factor_eigen_values = factorAnalyzed.get_eigenvalues()
# print(Original_Eigenvalues)
print("\nCommon Factor Eigenvalues: ", "\n", common_factor_eigen_values, "\n", "\nOriginal Eigenvalues: ", "\n", Original_Eigenvalues)
loadings = factorAnalyzed.loadings_
# print("\nLoadings: ", "\n", loadings, "\n", "Loadings Length: ", len(loadings), "\n","Loadings Shape: ", β.shape(loadings))
communalities = factorAnalyzed.get_communalities()
print("\nCommunalities: ", "\n", communalities, "\n")
factorVariances = factorAnalyzed.get_factor_variance()
variancesRowTitles = ['factor variances', 'proportional factor variances', 'cumulative factor variances']

factorVariances = pd.DataFrame.from_records(data=factorVariances, index=variancesRowTitles,\
     columns=['Factor ' + chr(i + ord('A')) for i in range(numberOfFactors)])
factorVariances.index.names = ['Variances']

uniquenesses = factorAnalyzed.get_uniquenesses()

loadingsTable = pd.DataFrame.from_records(data=loadings, index=transposedColumnTitles,\
     columns=['Factor ' + chr(i + ord('A')) for i in range(numberOfFactors)])
print("Loadings Table: ", "\n", loadingsTable)
# loadingsTable['Features'] = transposedColumnTitles
loadingsTable.index.names = ['Features']
loadingsTable['Communalities'] = communalities
print("Loadings Table: ", "\n", loadingsTable)
loadingsTable.info()
print(loadingsTable.head())
print(factorVariances)
# valuee2 = dataset.shape[1]+1
# valuee1 = dataset.shape[0]+1
# print("dataset.shape[1]+1: ", valuee2)
# print("dataset.shape[0]+1: ", valuee1)

print("Number of columns: ", len(dataset.columns))
print("Count of Original_Eigenvalues: ", len(Original_Eigenvalues))

# df = pd.DataFrame(β.random.random((5, 5)), columns=['thing ' + chr(i + ord('a')) for i in range(5)]) 
# print(df)

numberOfFeatures = len(dataset.columns)

plotter(numberOfFeatures, Original_Eigenvalues, filename, PlotSpot)

loadingsTable.to_csv("/Users/emmanueltoksadeniran/Desktop/Audio_Project/Output_Files/loadingsTable.csv")

# data = dataset.iloc[:,1:].copy()