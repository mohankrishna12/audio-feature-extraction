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


def plotter(attributes, CorrelationCoefficient, method, filename, PlotSpot):
    colorPalette = sns.color_palette("muted", n_colors=15)
    cmap = ListedColormap(sns.color_palette(colorPalette).as_hex())
    colors = β.random.randint(0,15,attributes)
    traceuse.scatter(CorrelationCoefficient, CorrelationCoefficient, c=colors, cmap=cmap)
    traceuse.title(str(method) + 'Correlation Coefficient')
    traceuse.ylabel('PCC Values')
    traceuse.yticks(fontsize=8, rotation=0)
    traceuse.xlabel('PCC Values')
    traceuse.xticks(fontsize=8, rotation=0)
    name = PlotSpot + str(filename) + ".pdf"
    traceuse.savefig(name, bbox_inches='tight')
    traceuse.show(block=False)
    traceuse.pause(2)
    traceuse.close("all")
    return


natural = '/Users/emmanueltoksadeniran/Desktop/Audio_Project/Input_Files/librosa_features_natural.csv'
artificial = '/Users/emmanueltoksadeniran/Desktop/Audio_Project/Input_Files/librosa_features_electronic.csv'
# natural = '/Users/emmanueltoksadeniran/Desktop/Audio_Project/Input_Files/few_librosa_features_natural.csv'
# artificial = '/Users/emmanueltoksadeniran/Desktop/Audio_Project/Input_Files/few_librosa_features_electronic.csv'
PlotSpot = '/Users/emmanueltoksadeniran/Desktop/Audio_Project/Output_Files/Charts/'

inPersonFile = open(natural, 'r')
inPerson = list(csv.reader(inPersonFile, delimiter=','))
features = [str(x) for x in inPerson[0]]

inPerson = inPerson[1:]
inPerson = list(filter(valid, inPerson))
inPerson = β.asarray(inPerson, dtype=β.float64)
rowLength = [row[0] for row in inPerson]
length = len(rowLength)
filename = os.path.basename(natural)
print ("\n")
print("The ingested file named " + str(filename) + " consists of "+ str(length) + " entries/variables (rows) and " \
+str(len(features)) + " attributes/samples (columns)" )
number = len(features)
print("inPerson's Shape: ", β.shape(inPerson), "\n")


electronicFile = open(artificial, 'r')
electronic = list(csv.reader(electronicFile, delimiter=','))
attributes = [str(x) for x in electronic[0]]

electronic = electronic[1:]
electronic = list(filter(valid, electronic))
electronic = β.asarray(electronic, dtype=β.float64)
rowLengths = [row[0] for row in electronic]
lengths = len(rowLengths)
filename = os.path.basename(artificial)
print("The ingested file named " + str(filename) + " consists of "+ str(lengths) + " entries/variables (rows) and " \
+str(len(attributes)) + " attributes/samples (columns)" )
count = len(attributes)
print("Electronic's Shape: ", β.shape(electronic), "\n")



MIC = "MIC "
filenames = "MIC_Chart"
α = 15
mic_p, tic_p =  pstats(inPerson, alpha=α, c=5, est="mic_e")
bivariateDependence, tic_c =  cstats(inPerson, electronic, alpha=α, c=5, est="mic_e")
print ("Pair-wise MIC (inPerson vs. electronic):\n")
print (bivariateDependence, "\n")
print("Pair-wise MIC's Matrix Shape: ", β.shape(bivariateDependence), "\n")
MIClength = len(bivariateDependence)**2



bivariateDependencePandas = pd.DataFrame(bivariateDependence)
bivariateDependencePandasPearson = bivariateDependencePandas.corr(method='pearson')
print("Pair-wise Pandas Pearson's Matrix:", "\n", bivariateDependencePandasPearson, "\n",\
     "Pair-wise Pandas Pearson's Matrix Shape:", β.shape(bivariateDependencePandasPearson), "\n")

bivariateDependencePandasKendall = bivariateDependencePandas.corr(method='kendall')
print("Pair-wise Pandas Kendall's Matrix:", "\n", bivariateDependencePandasKendall, "\n", \
    "Pair-wise Pandas Kendall's Matrix Shape:", β.shape(bivariateDependencePandasKendall), "\n")

bivariateDependencePandasSpearman = bivariateDependencePandas.corr(method='spearman')
print("Pair-wise Pandas Spearman's Matrix:", "\n", bivariateDependencePandasSpearman, "\n", \
    "Pair-wise Pandas Spearman's Matrix Shape:", β.shape(bivariateDependencePandasSpearman), "\n")



Pearson = "Pearson "
filename = "PCC_Chart"
attribute = len(attributes)
N = electronic.shape[0]
sA = inPerson.sum(0)
sB = electronic.sum(0)
p1 = N*β.einsum('ij,ik->kj',inPerson,electronic)
p2 = sA*sB[:,None]
p3 = N*((electronic**2).sum(0)) - (sB**2)
p4 = N*((inPerson**2).sum(0)) - (sA**2)
pcorr = ((p1 - p2)/β.sqrt(p4*p3[:,None]))
PearsonCorrelationCoefficient = pcorr[β.nanargmax(β.abs(pcorr),axis=0),β.arange(pcorr.shape[1])]
print("Pearson's Column-wise Correlation Coefficient Matrix:  ", "\n",PearsonCorrelationCoefficient,\
     "\n", "PCC's Matrix Shape: ", β.shape(PearsonCorrelationCoefficient), "\n")
attributes = β.asarray(attributes)
stackedLabelsPCC = β.stack((attributes, PearsonCorrelationCoefficient))
stackedLabelsPCC = stackedLabelsPCC.tolist()



ColumnWise = "ColumnWise "
filenaming = "ColumnWise"
datasetOne = pd.DataFrame(inPerson) 
datasetTwo = pd.DataFrame(electronic)
correlsSeries = datasetOne.corrwith(datasetTwo, axis = 0)
correlsSeries = β.asarray(correlsSeries, dtype=β.float64)
print("Column-wise Coefficent's Matrix in Pandas:", "\n", correlsSeries, "\n", \
    "Column-wise Pandas Pearson's Matrix Shape:", β.shape(correlsSeries), "\n")

stackedLabelsSCC = β.stack((attributes, correlsSeries))
stackedLabelsSCC = stackedLabelsSCC.tolist()



plotter(attribute, PearsonCorrelationCoefficient, Pearson, filename, PlotSpot)
plotter(attribute, correlsSeries, ColumnWise, filenaming, PlotSpot)
plotter(MIClength, bivariateDependence, MIC, filenames, PlotSpot)


β.savetxt("/Users/emmanueltoksadeniran/Desktop/Audio_Project/Output_Files/MIC.csv", bivariateDependence, delimiter=",")
β.savetxt("/Users/emmanueltoksadeniran/Desktop/Audio_Project/Output_Files/Pearson Correlation Coefficient.csv", bivariateDependencePandasPearson, delimiter=",")
β.savetxt("/Users/emmanueltoksadeniran/Desktop/Audio_Project/Output_Files/Kendall Correlation Coefficient.csv", bivariateDependencePandasKendall, delimiter=",")
β.savetxt("/Users/emmanueltoksadeniran/Desktop/Audio_Project/Output_Files/Spearman Correlation Coefficient.csv", bivariateDependencePandasSpearman, delimiter=",")
β.savetxt("/Users/emmanueltoksadeniran/Desktop/Audio_Project/Output_Files/PearsonColumns.csv", stackedLabelsPCC, delimiter=",", fmt="%s")
β.savetxt("/Users/emmanueltoksadeniran/Desktop/Audio_Project/Output_Files/PandasColumns.csv", stackedLabelsSCC, delimiter=",", fmt="%s")

# β.savetxt("/Users/emmanueltoksadeniran/Desktop/Audio_Project/Output_Files/smallMICs.csv", bivariateDependence, delimiter=",")
# β.savetxt("/Users/emmanueltoksadeniran/Desktop/Audio_Project/Output_Files/smallPearsons.csv", stackedLabelsPCC, delimiter=",", fmt="%s")
# β.savetxt("/Users/emmanueltoksadeniran/Desktop/Audio_Project/Output_Files/smallColumns.csv", stackedLabelsSCC, delimiter=",", fmt="%s")

