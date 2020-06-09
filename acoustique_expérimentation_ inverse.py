import numpy as β
from minepy import pstats, cstats, MINE
import matplotlib.pyplot as traceuse
import csv
from itertools import product
from matplotlib.ticker import MaxNLocator
from array import array
import sys, os
import pandas as pd


def valid(x):
    if '' in x:
        return False
    else:
        return True

# attributesSelected = 5

natural = '/Users/emmanueltoksadeniran/Desktop/Audio_Project/librosa_features_natural.csv'
artificial = '/Users/emmanueltoksadeniran/Desktop/Audio_Project/librosa_features_electronic.csv'
# natural = '/Users/emmanueltoksadeniran/Desktop/Audio_Project/few_librosa_features_natural.csv'
# artificial = '/Users/emmanueltoksadeniran/Desktop/Audio_Project/few_librosa_features_electronic.csv'

inPersonFile = open(natural, 'r')
inPerson = list(csv.reader(inPersonFile, delimiter=','))
# types = ['f8', 'f8', 'U50', 'i4', 'i4', 'i4', 'i4', 'i4']
# inPerson = β.genfromtxt(natural, dtype=types, delimiter=',', names=True)
features = [str(x) for x in inPerson[0]]

inPerson = inPerson[1:]
inPerson = list(filter(valid, inPerson))
# print("inPerson Data: ", inPerson, "Type: ", type(inPerson), "\n")
inPerson = β.asarray(inPerson, dtype=β.float64)
print("inPerson Data: ", inPerson, "Type: ", type(inPerson), "\n")
rowLength = [row[0] for row in inPerson]
length = len(rowLength)
filename = os.path.basename(natural)
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
print("Electronic Data: ", electronic, "Type: ", type(electronic), "\n")
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
print ("\n")
print("Pair-wise MIC's Matrix Shape: ", β.shape(bivariateDependence), "\n")


bivariateDependencePandas = pd.DataFrame(bivariateDependence)
# bivariateDependencePandas.head()
print("Pair-wise MIC's Matrix in Pandas:", "\n", bivariateDependencePandas, "\n", "Pair-wise Pandas Pearson's Matrix Shape:", β.shape(bivariateDependencePandas), "\n")
bivariateDependencePandas = bivariateDependencePandas.corr(method='pearson')
print("Pair-wise Pandas Pearson's Matrix:", "\n", bivariateDependencePandas, "\n", "Pair-wise Pandas Pearson's Matrix Shape:", β.shape(bivariateDependencePandas), "\n")


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


# corr = []
# for i in range(len(inPerson)):
#     for j in range(len(electronic)-i):
#         corr.extend(β.correlate(inPerson[i], electronic[j+i]))
# corr_avg = β.average(corr)
# print(corr_avg, "\n")
# print (" ".join(map(str, corr)), "\n")
# print("Numpy's Matrix Shape: ", β.shape(corr), "\n")


# sumOfColumns = [sum(columns) for columns in zip(*bivariateDependence)]
# print("Row of the Sum of Columns: ", sumOfColumns, "Type: ", type(sumOfColumns), "\n")
# print("Row of the Sum of Columns's Shape: ", β.shape(sumOfColumns), "\n")

# matrixDenominator = lengths * [lengths]
# print("Number of rows: ", lengths)
# print("Number of columns: ", lengths)
# print("Denominator for Averaging: ", matrixDenominator)
# μSumOfStrengths = β.divide(sumOfColumns, matrixDenominator)
# print("Average Sum of Errors: ", μSumOfStrengths)

β.savetxt("/Users/emmanueltoksadeniran/Desktop/Audio_Project/MIC.csv", bivariateDependence, delimiter=",")
β.savetxt("/Users/emmanueltoksadeniran/Desktop/Audio_Project/PCC.csv", PearsonCorrelationCoefficient, delimiter=",")

