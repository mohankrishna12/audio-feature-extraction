import numpy as β
from minepy import pstats, cstats, MINE
import matplotlib.pyplot as traceuse
import csv
from itertools import product
from matplotlib.ticker import MaxNLocator
from array import array
from array import *
import sys, os
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap


def valid(x):
    if '' in x:
        return False
    else:
        return True


natural = '/Users/emmanueltoksadeniran/Desktop/Audio_Project/Input_Files/librosa_features_natural.csv'
artificial = '/Users/emmanueltoksadeniran/Desktop/Audio_Project/Input_Files/librosa_features_electronic.csv'
# natural = '/Users/emmanueltoksadeniran/Desktop/Audio_Project/Input_Files/few_librosa_features_natural.csv'
# artificial = '/Users/emmanueltoksadeniran/Desktop/Audio_Project/Input_Files/few_librosa_features_electronic.csv'

inPersonFile = open(natural, 'r')
inPerson = list(csv.reader(inPersonFile, delimiter=','))
features = [str(x) for x in inPerson[0]]

inPerson = inPerson[1:]
inPerson = list(filter(valid, inPerson))
# print("inPerson Data: ", inPerson, "Type: ", type(inPerson), "\n")
inPerson = β.asarray(inPerson, dtype=β.float64)
# print("inPerson Data: ", inPerson, "Type: ", type(inPerson), "\n")
rowLength = [row[0] for row in inPerson]
length = len(rowLength)
filename = os.path.basename(natural)
print ("\n")
print("The ingested file named " + str(filename) + " consists of "+ str(length) + " entries/variables (rows) and " \
+str(len(features)) + " attributes/samples (columns)" )
number = len(features)
# print("Number of features per row/entry: ", number)
# print("List of features per row/entry: ", features)
print("inPerson's Shape: ", β.shape(inPerson), "\n")


electronicFile = open(artificial, 'r')
electronic = list(csv.reader(electronicFile, delimiter=','))
attributes = [str(x) for x in electronic[0]]

electronic = electronic[1:]
electronic = list(filter(valid, electronic))
# print("Electronic Data: ", electronic, "Type: ", type(electronic), "\n")
electronic = β.asarray(electronic, dtype=β.float64)
# print("Electronic Data: ", electronic, "Type: ", type(electronic), "\n")
rowLengths = [row[0] for row in electronic]
lengths = len(rowLengths)
filename = os.path.basename(artificial)
print("The ingested file named " + str(filename) + " consists of "+ str(lengths) + " entries/variables (rows) and " \
+str(len(attributes)) + " attributes/samples (columns)" )
count = len(attributes)
# print("Number of features per row/entry: ", count)
# print("List of features per row/entry: ", attributes)
print("Electronic's Shape: ", β.shape(electronic), "\n")

α = 15
mic_p, tic_p =  pstats(inPerson, alpha=α, c=5, est="mic_e")
bivariateDependence, tic_c =  cstats(inPerson, electronic, alpha=α, c=5, est="mic_e")
print ("Pair-wise MIC (inPerson vs. electronic):\n")
print (bivariateDependence, "\n")
print("Pair-wise MIC's Matrix Shape: ", β.shape(bivariateDependence), "\n")


bivariateDependencePandas = pd.DataFrame(bivariateDependence)
# bivariateDependencePandas.head()
# print("Pair-wise MIC's Matrix in Pandas:", "\n", bivariateDependencePandas, "\n", "Pair-wise Pandas Pearson's Matrix Shape:", β.shape(bivariateDependencePandas), "\n")
bivariateDependencePandasPearson = bivariateDependencePandas.corr(method='pearson')
print("Pair-wise Pandas Pearson's Matrix:", "\n", bivariateDependencePandasPearson, "\n",\
     "Pair-wise Pandas Pearson's Matrix Shape:", β.shape(bivariateDependencePandasPearson), "\n")

bivariateDependencePandasKendall = bivariateDependencePandas.corr(method='kendall')
print("Pair-wise Pandas Kendall's Matrix:", "\n", bivariateDependencePandasKendall, "\n", \
    "Pair-wise Pandas Kendall's Matrix Shape:", β.shape(bivariateDependencePandasKendall), "\n")

bivariateDependencePandasSpearman = bivariateDependencePandas.corr(method='spearman')
print("Pair-wise Pandas Spearman's Matrix:", "\n", bivariateDependencePandasSpearman, "\n", \
    "Pair-wise Pandas Spearman's Matrix Shape:", β.shape(bivariateDependencePandasSpearman), "\n")

N = electronic.shape[0]
sA = inPerson.sum(0)
sB = electronic.sum(0)
p1 = N*β.einsum('ij,ik->kj',inPerson,electronic)
p2 = sA*sB[:,None]
p3 = N*((electronic**2).sum(0)) - (sB**2)
p4 = N*((inPerson**2).sum(0)) - (sA**2)
pcorr = ((p1 - p2)/β.sqrt(p4*p3[:,None]))
PearsonCorrelationCoefficient = pcorr[β.nanargmax(β.abs(pcorr),axis=0),β.arange(pcorr.shape[1])]
print("Pearson'Pair-wise Correlation Coefficient Matrix:  ", "\n",PearsonCorrelationCoefficient, "\n", "PCC's Matrix Shape: ", β.shape(PearsonCorrelationCoefficient), "\n")
attributes = β.asarray(attributes)
# print("Label Matrix:  ", "\n",attributes, "\n", "Label's Matrix Shape: ", β.shape(attributes), "\n", "Label's Type", type(attributes), "\n")
stackedLabelsPCC = β.stack((attributes, PearsonCorrelationCoefficient))
# print("Attributed Matrix:  ", "\n",stackedLabelsPCC, "\n", "Attributed Matrix's Shape: ", β.shape(stackedLabelsPCC), "\n", "Label's Type", type(stackedLabelsPCC), "\n")

# stackedLabelsPCC = stackedLabelsPCC.tolist()
# print("Attributed Matrix:  ", "\n",stackedLabelsPCC, "\n", "Attributed Matrix's Shape: ", β.shape(stackedLabelsPCC), "\n", "Label's Type", type(stackedLabelsPCC), "\n")


# corr = []
# for i in range(len(inPerson)):
#     for j in range(len(electronic)-i):
#         corr.extend(β.correlate(inPerson[i], electronic[j+i]))
# corr_avg = β.average(corr)
# print(corr_avg, "\n")
# print (" ".join(map(str, corr)), "\n")
# print("Numpy's Matrix Shape: ", β.shape(corr), "\n")

MatrixPlot = '/Users/emmanueltoksadeniran/Desktop/Audio_Project/Output_Files/Charts/'

# plotted = traceuse.matshow(stackedLabelsPCC, cmap='gist_earth')
# ax = sns.heatmap(stackedLabelsPCC)
# stableAxis = β.ones((len(attributes)))
# print("Stable Axis Matrix:  ", "\n",stableAxis) 
# plotted = traceuse.scatter(stableAxis, PearsonCorrelationCoefficient, color='chartreuse', linestyle='-', marker='+', label='X data')
# plotted = traceuse.scatter(PearsonCorrelationCoefficient, PearsonCorrelationCoefficient, color='fuchsia', linestyle='-', marker='+', label='X data')
# plotted = traceuse.bar(attributes, PearsonCorrelationCoefficient)

MIClength = len(bivariateDependence)**2

colorPalette = sns.color_palette("muted", n_colors=15)
cmap = ListedColormap(sns.color_palette(colorPalette).as_hex())
colors = β.random.randint(0,15,MIClength)
traceuse.scatter(bivariateDependence, bivariateDependence, c=colors, cmap=cmap)
traceuse.title('Correlation Coefficient')
traceuse.ylabel('MIC Values')
traceuse.yticks(fontsize=8, rotation=0)
traceuse.xlabel('MIC Values')
traceuse.xticks(fontsize=8, rotation=0)
filename = "MIC_Chart"
name = MatrixPlot + str(filename) + ".pdf"
traceuse.savefig(name, bbox_inches='tight')
traceuse.show(block=False)
traceuse.pause(2)
traceuse.close("all")

colorPalette = sns.color_palette("muted", n_colors=15)
cmap = ListedColormap(sns.color_palette(colorPalette).as_hex())
colors = β.random.randint(0,15,len(attributes))
traceuse.scatter(PearsonCorrelationCoefficient, PearsonCorrelationCoefficient, c=colors, cmap=cmap)
traceuse.title('Correlation Coefficient')
traceuse.ylabel('PCC Values')
traceuse.yticks(fontsize=8, rotation=0)
traceuse.xlabel('PCC Values')
traceuse.xticks(fontsize=8, rotation=0)
filename = "PCC_Chart"
name = MatrixPlot + str(filename) + ".pdf"
traceuse.savefig(name, bbox_inches='tight')
traceuse.show(block=False)
traceuse.pause(2)
traceuse.close("all")


β.savetxt("/Users/emmanueltoksadeniran/Desktop/Audio_Project/Output_Files/MIC.csv", bivariateDependence, delimiter=",")
β.savetxt("/Users/emmanueltoksadeniran/Desktop/Audio_Project/Output_Files/Pearson Correlation Coefficient.csv", bivariateDependencePandasPearson, delimiter=",")
β.savetxt("/Users/emmanueltoksadeniran/Desktop/Audio_Project/Output_Files/Kendall Correlation Coefficient.csv", bivariateDependencePandasKendall, delimiter=",")
β.savetxt("/Users/emmanueltoksadeniran/Desktop/Audio_Project/Output_Files/Pearson Correlation Coefficient.csv", bivariateDependencePandasSpearman, delimiter=",")
β.savetxt("/Users/emmanueltoksadeniran/Desktop/Audio_Project/Output_Files/Spearman.csv", stackedLabelsPCC, delimiter=",", fmt="%s")

# β.savetxt("/Users/emmanueltoksadeniran/Desktop/Audio_Project/Output_Files/smallMIC.csv", bivariateDependence, delimiter=",")
# β.savetxt("/Users/emmanueltoksadeniran/Desktop/Audio_Project/Output_Files/smallPCC.csv", stackedLabelsPCC, delimiter=",", fmt="%s")

