import sklearn as scikit
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import linear_model
import math

'''Possibly cities with closest stats
0.0 0.0
[1993    0 1993]
848.0 983.0
[ 848    0 1993]
595.0 1893.0
[ 595    0 1993]
389.0 1111.0
[ 389    0 1993]
1033.0 1224.0
[1033    0 1993]
27.0 1473.0
[  27    0 1993]
668.0 699.0
[ 668    0 1993]
1658.0 1926.0
[1658    0 1993]
1461.0 1768.0
[1461    0 1993]
1171.0 1413.0
[1171    0 1993]
'''
mostCloseCities = [(848,983),(595,1893),(389,1111),(1033,1224),(27,1473),(668,699),(1658,1926),(1461,1768),(1171,1413)]

'''Possibly most different cities
1134.0 1615.0
[1134 1993 1993]
139.0 1134.0
[ 139 1993 1993]
910.0 1134.0
[ 910 1993 1993]
849.0 1134.0
[ 849 1993 1993]
1158.0 1847.0
[1158 1993 1993]
287.0 1134.0
[ 287 1993 1993]
657.0 1134.0
[ 657 1993 1993]
744.0 1134.0
[ 744 1993 1993]
492.0 1134.0
[ 492 1993 1993]
1500.0 1847.0
[1500 1993 1993]
'''
mostDifCities = [(1134,1615),(139,1134),(849,1134),(1158,1847),(287,1134),(657,1134),(744,1134),(744,1134),(492,1134),(1500,1847)]

def Num_Rows(matrix):
    return matrix.shape[0]

def Num_Columns(matrix):
    return matrix.shape[1]
    

#function to calc Linear Regression for all columns in 'matrix' to fit to tarVec and return dictionary of all names of columns with their parameters
def calcLinearReg(matrix,tarVec,ResidualFlag):
    header = matrix.columns
    correctDimX = np.empty(shape=(Num_Rows(matrix),1), dtype=float)
    correctDimZ = np.empty(shape=(Num_Rows(matrix),1), dtype=float)
    linModel = linear_model.LinearRegression(fit_intercept=True) #May want to be False need to look at what I pass in
    for i in range(0,Num_Columns(matrix)):
        for k in range(0,Num_Rows(matrix)):
            correctDimX[k,0] = matrix.iloc[k,i]
            correctDimZ[k,0] = tarVec.iloc[k]
        linModelAct = linModel.fit(correctDimX, correctDimZ)
        linPredict = linModelAct.predict(correctDimZ)
        if i == 0:
            if ResidualFlag:
                 newMatrix = pd.DataFrame(abs(linPredict-correctDimX))
            else:
                newMatrix = pd.DataFrame(linPredict)
        else:
            if ResidualFlag:
                newMatrix.insert(i, header[i], abs(linPredict-correctDimX))
            else:
                newMatrix.insert(i, header[i], linPredict)
    newMatrix.rename(columns={0:header[0]}, inplace=True)
    avgVec = newMatrix.mean(0)
    return matrix


#calculates ratio of white to PoC in city
def modifyRace(matrix,header):
    newRaceMetric = 0
    newRaceArray = np.empty(shape= (Num_Rows(matrix),1), dtype=float)
    for i in range(0,Num_Rows(matrix)):
        percentWhite = matrix.iloc[i,3]
        percentBlack = matrix.iloc[i,2]
        percentAsian = matrix.iloc[i,4]
        percentHispanic = matrix.iloc[i,5]
        newRaceMetric = percentWhite / ( percentBlack+percentHispanic+percentAsian)
        newRaceArray[i] = newRaceMetric
    matrix = matrix[header]
    return matrix, newRaceArray
    

def correlWReversePCA(PcaMatrix, TargetVector ):
    pcaNumColumns = Num_Columns(PcaMatrix)
    pcaNumRows = Num_Rows(PcaMatrix)
    nMinusOne = (pcaNumRows-1)

    stdPca = PcaMatrix.std(axis=0)
    stdTar = TargetVector.std(axis=0)
    stdPca = stdPca * stdTar
    
    tarVecMean = np.mean(TargetVector)
    TargetVector -= tarVecMean

    for i in range(0,pcaNumRows):
        PcaMatrix.iloc[i] = PcaMatrix.iloc[i] * TargetVector[i]
    
    sumPca = PcaMatrix.sum(axis=0)

    

    for i in range(0,pcaNumColumns):
        sumPca[i] = sumPca[i]/(stdPca[i])/nMinusOne

    return sumPca










# From https://archive.ics.uci.edu/ml/datasets/communities+and+crime#::text=UCI%20Machine%20Learning%20Repository%3A%20Communities%20and%20Crime%20Data%20Set&text=Abstract%3A%20Communities%20within%20the%20United,from%20the%201995%20FBI%20UCR
# Min Max Mean SD Correl Median Mode Missing 
summary = {"population": {"Min": 0, "Max": 1, "Mean": 0.06, "SD": 0.13, "Correl": 0.37, "Median": 0.02, "Mode": 0.01, "Missing": 0}, 
           "householdsize": {"Min": 0, "Max": 1, "Mean": 0.46, "SD": 0.16, "Correl": -0.03, "Median": 0.44, "Mode": 0.41, "Missing": 0 },
           "racepctblack": {"Min": 0, "Max": 1, "Mean": 0.18, "SD": 0.25, "Correl": 0.63, "Median": 0.06, "Mode": 0.01, "Missing": 0 },
           "racePctWhite": {"Min": 0, "Max": 1, "Mean": 0.75, "SD": 0.24, "Correl": -0.68, "Median": 0.85, "Mode": 0.98, "Missing": 0 },
           "racePctAsian": {"Min": 0, "Max": 1, "Mean": 0.15, "SD": 0.21, "Correl": 0.04, "Median": 0.07, "Mode": 0.02, "Missing": 0 },
           "racePctHisp": {"Min": 0, "Max": 1, "Mean": 0.14, "SD": 0.23, "Correl": 0.29, "Median": 0.04, "Mode": 0.01, "Missing": 0 },
           "agePct12t21": {"Min": 0, "Max": 1, "Mean": 0.42, "SD": 0.16, "Correl": 0.06, "Median": 0.4, "Mode": 0.38, "Missing": 0 },
           "agePct12t29": {"Min": 0, "Max": 1, "Mean": 0.49, "SD": 0.14, "Correl": 0.15, "Median": 0.48, "Mode": 0.49, "Missing": 0 },
           "agePct16t24": {"Min": 0, "Max": 1, "Mean": 0.34, "SD": 0.17, "Correl": 0.10, "Median": 0.29, "Mode": 0.29, "Missing": 0 },
           "agePct65up": {"Min": 0, "Max": 1, "Mean": 0.42, "SD": 0.18, "Correl": 0.07, "Median": 0.42, "Mode": 0.47, "Missing": 0 },
           "numbUrban": {"Min": 0, "Max": 1, "Mean": 0.06, "SD": 0.13, "Correl": 0.36, "Median": 0.03, "Mode": 0, "Missing": 0 },
           "pctUrban": {"Min": 0, "Max": 1, "Mean": 0.70, "SD": 0.44, "Correl": 0.08, "Median": 1, "Mode": 1, "Missing": 0 },
           "medIncome": {"Min": 0, "Max": 1, "Mean": 0.36, "SD": 0.21, "Correl": -0.42, "Median": 0.32, "Mode": 0.23, "Missing": 0 },
           "pctWWage": {"Min": 0, "Max": 1, "Mean": 0.56, "SD": 0.18, "Correl": -0.31, "Median": 0.56, "Mode": 0.58, "Missing": 0 },
           "pctWFarmSelf": {"Min": 0, "Max": 1, "Mean": 0.29, "SD": 0.20, "Correl": -0.15, "Median": 0.23, "Mode": 0.16, "Missing": 0 },
           "pctWInvInc": {"Min": 0, "Max": 1, "Mean": 0.50, "SD": 0.18, "Correl": -0.58, "Median": 0.48, "Mode": 0.41, "Missing": 0 },
           "pctWSocSec": {"Min": 0, "Max": 1, "Mean": 0.47, "SD": 0.17, "Correl": 0.12, "Median": 0.475, "Mode": 0.56, "Missing": 0 },
           "pctWPubAsst": {"Min": 0, "Max": 1, "Mean": 0.32, "SD": 0.22, "Correl": 0.57, "Median": 0.26, "Mode": 0.1, "Missing": 0 },
           "pctWRetire": {"Min": 0, "Max": 1, "Mean": 0.48, "SD": 0.17, "Correl": -0.10, "Median": 0.47, "Mode": 0.44, "Missing": 0 },
           "medFamInc": {"Min": 0, "Max": 1, "Mean": 0.38, "SD": 0.20, "Correl": -0.44, "Median": 0.33, "Mode": 0.25, "Missing": 0 },
           "perCapInc": {"Min": 0, "Max": 1, "Mean": 0.35, "SD": 0.19, "Correl": -0.35, "Median": 0.3, "Mode": 0.23, "Missing": 0 },
           "whitePerCap": {"Min": 0, "Max": 1, "Mean": 0.37, "SD": 0.19, "Correl": -0.21, "Median": 0.32, "Mode": 0.3, "Missing": 0 },
           "blackPerCap": {"Min": 0, "Max": 1, "Mean": 0.29, "SD": 0.17, "Correl": -0.28, "Median": 0.25, "Mode": 0.18, "Missing": 0 },
           "indianPerCap": {"Min": 0, "Max": 1, "Mean": 0.20, "SD": 0.16, "Correl": -0.09, "Median": 0.17, "Mode": 0, "Missing": 0 },
           "AsianPerCap": {"Min": 0, "Max": 1, "Mean": 0.32, "SD": 0.20, "Correl": -0.16, "Median": 0.28, "Mode": 0.18, "Missing": 0 },
           "OtherPerCap": {"Min": 0, "Max": 1, "Mean": 0.28, "SD": 0.19, "Correl": -0.13, "Median": 0.25, "Mode": 0, "Missing": 1 },
           "HispPerCap": {"Min": 0, "Max": 1, "Mean": 0.39, "SD": 0.18, "Correl": -0.24, "Median": 0.345, "Mode": 0.3, "Missing": 0 },
           "NumUnderPov": {"Min": 0, "Max": 1, "Mean": 0.06, "SD": 0.13, "Correl": 0.45, "Median": 0.02, "Mode": 0.01, "Missing": 0 },
           "PctPopUnderPov": {"Min": 0, "Max": 1, "Mean": 0.30, "SD": 0.23, "Correl": 0.52, "Median": 0.25, "Mode": 0.08, "Missing": 0 },
           "PctLess9thGrade": {"Min": 0, "Max": 1, "Mean": 0.32, "SD": 0.21, "Correl": 0.41, "Median": 0.27, "Mode": 0.19, "Missing": 0 },
           "PctNotHSGrad": {"Min": 0, "Max": 1, "Mean": 0.38, "SD": 0.20, "Correl": 0.48, "Median": 0.36, "Mode": 0.39, "Missing": 0 },
           "PctBSorMore": {"Min": 0, "Max": 1, "Mean": 0.36, "SD": 0.21, "Correl": -0.31, "Median": 0.31, "Mode": 0.18, "Missing": 0 },
           "PctUnemployed": {"Min": 0, "Max": 1, "Mean": 0.36, "SD": 0.20, "Correl": 0.50, "Median": 0.32, "Mode": 0.24, "Missing": 0 },
           "PctEmploy": {"Min": 0, "Max": 1, "Mean": 0.50, "SD": 0.17, "Correl": -0.33, "Median": 0.51, "Mode": 0.56, "Missing": 0 },
           "PctEmplManu": {"Min": 0, "Max": 1, "Mean": 0.40, "SD": 0.20, "Correl": -0.04, "Median": 0.37, "Mode": 0.26, "Missing": 0 },
           "PctEmplProfServ": {"Min": 0, "Max": 1, "Mean": 0.44, "SD": 0.18, "Correl": -0.07, "Median": 0.41, "Mode": 0.36, "Missing": 0 },
           "PctOccupManu": {"Min": 0, "Max": 1, "Mean": 0.39, "SD": 0.20, "Correl": 0.30, "Median": 0.37, "Mode": 0.32, "Missing": 0 },
           "PctOccupMgmtProf": {"Min": 0, "Max": 1, "Mean": 0.44, "SD": 0.19, "Correl": -0.34, "Median": 0.4, "Mode": 0.36, "Missing": 0 },
           "MalePctDivorce": {"Min": 0, "Max": 1, "Mean": 0.46, "SD": 0.18, "Correl": 0.53, "Median": 0.47, "Mode": 0.56, "Missing": 0 },
           "MalePctNevMarr": {"Min": 0, "Max": 1, "Mean": 0.43, "SD": 0.18, "Correl": 0.30, "Median": 0.4, "Mode": 0.38, "Missing": 0 },
           "FemalePctDiv": {"Min": 0, "Max": 1, "Mean": 0.49, "SD": 0.18, "Correl": 0.56, "Median": 0.5, "Mode": 0.54, "Missing": 0 },
           "TotalPctDiv": {"Min": 0, "Max": 1, "Mean": 0.49, "SD": 0.18, "Correl": 0.55, "Median": 0.5, "Mode": 0.57, "Missing": 0 },
           "PersPerFam": {"Min": 0, "Max": 1, "Mean": 0.49, "SD": 0.15, "Correl": 0.14, "Median": 0.47, "Mode": 0.44, "Missing": 0 },
           "PctFam2Par": {"Min": 0, "Max": 1, "Mean": 0.61, "SD": 0.20, "Correl": -0.71, "Median": 0.63, "Mode": 0.7, "Missing": 0 },
           "PctKids2Par": {"Min": 0, "Max": 1, "Mean": 0.62, "SD": 0.21, "Correl": -0.74, "Median": 0.64, "Mode": 0.72, "Missing": 0 },
           "PctYoungKids2Par": {"Min": 0, "Max": 1, "Mean": 0.66, "SD": 0.22, "Correl": -0.67, "Median": 0.7, "Mode": 0.91, "Missing": 0 },
           "PctTeen2Par": {"Min": 0, "Max": 1, "Mean": 0.58, "SD": 0.19, "Correl": -0.66, "Median": 0.61, "Mode": 0.6, "Missing": 0 },
           "PctWorkMomYoungKids": {"Min": 0, "Max": 1, "Mean": 0.50, "SD": 0.17, "Correl": -0.02, "Median": 0.51, "Mode": 0.51, "Missing": 0 },
           "PctWorkMom": {"Min": 0, "Max": 1, "Mean": 0.53, "SD": 0.18, "Correl": -0.15, "Median": 0.54, "Mode": 0.57, "Missing": 0 },
           "NumIlleg": {"Min": 0, "Max": 1, "Mean": 0.04, "SD": 0.11, "Correl": 0.47, "Median": 0.01, "Mode": 0, "Missing": 0 },
           "PctIlleg": {"Min": 0, "Max": 1, "Mean": 0.25, "SD": 0.23, "Correl": 0.74, "Median": 0.17, "Mode": 0.09, "Missing": 0 },
           "NumImmig": {"Min": 0, "Max": 1, "Mean": 0.03, "SD": 0.09, "Correl": 0.29, "Median": 0.01, "Mode": 0, "Missing": 0 },
           "PctImmigRecent": {"Min": 0, "Max": 1, "Mean": 0.32, "SD": 0.22, "Correl": 0.17, "Median": 0.29, "Mode": 0, "Missing": 0 },
           "PctImmigRec5": {"Min": 0, "Max": 1, "Mean": 0.36, "SD": 0.21, "Correl": 0.22, "Median": 0.34, "Mode": 0, "Missing": 0 },
           "PctImmigRec8": {"Min": 0, "Max": 1, "Mean": 0.40, "SD": 0.20, "Correl": 0.25, "Median": 0.39, "Mode": 0.26, "Missing": 0 },
           "PctImmigRec10": {"Min": 0, "Max": 1, "Mean": 0.43, "SD": 0.19, "Correl": 0.29, "Median": 0.43, "Mode": 0.43, "Missing": 0 },
           "PctRecentImmig": {"Min": 0, "Max": 1, "Mean": 0.18, "SD": 0.24, "Correl": 0.23, "Median": 0.09, "Mode": 0.01, "Missing": 0 },
           "PctRecImmig5": {"Min": 0, "Max": 1, "Mean": 0.18, "SD": 0.24, "Correl": 0.25, "Median": 0.08, "Mode": 0.02, "Missing": 0 },
           "PctRecImmig8": {"Min": 0, "Max": 1, "Mean": 0.18, "SD": 0.24, "Correl": 0.25, "Median": 0.09, "Mode": 0.02, "Missing": 0 },
           "PctRecImmig10": {"Min": 0, "Max": 1, "Mean": 0.18, "SD": 0.23, "Correl": 0.26, "Median": 0.09, "Mode": 0.02, "Missing": 0 },
           "PctSpeakEnglOnly": {"Min": 0, "Max": 1, "Mean": 0.79, "SD": 0.23, "Correl": -0.24, "Median": 0.87, "Mode": 0.96, "Missing": 0 },
           "PctNotSpeakEnglWell": {"Min": 0, "Max": 1, "Mean": 0.15, "SD": 0.22, "Correl": 0.30, "Median": 0.06, "Mode": 0.03, "Missing": 0 },
           "PctLargHouseFam": {"Min": 0, "Max": 1, "Mean": 0.27, "SD": 0.20, "Correl": 0.38, "Median": 0.2, "Mode": 0.17, "Missing": 0 },
           "PctLargHouseOccup": {"Min": 0, "Max": 1, "Mean": 0.25, "SD": 0.19, "Correl": 0.29, "Median": 0.19, "Mode": 0.19, "Missing": 0 },
           "PersPerOccupHous": {"Min": 0, "Max": 1, "Mean": 0.46, "SD": 0.17, "Correl": -0.04, "Median": 0.44, "Mode": 0.37, "Missing": 0 },
           "PersPerOwnOccHous": {"Min": 0, "Max": 1, "Mean": 0.49, "SD": 0.16, "Correl": -0.12, "Median": 0.48, "Mode": 0.45, "Missing": 0 },
           "PersPerRentOccHous": {"Min": 0, "Max": 1, "Mean": 0.40, "SD": 0.19, "Correl": 0.25, "Median": 0.36, "Mode": 0.32, "Missing": 0 },
           "PctPersOwnOccup": {"Min": 0, "Max": 1, "Mean": 0.56, "SD": 0.20, "Correl": -0.53, "Median": 0.56, "Mode": 0.54, "Missing": 0 },
           "PctPersDenseHous": {"Min": 0, "Max": 1, "Mean": 0.19, "SD": 0.21, "Correl": 0.45, "Median": 0.11, "Mode": 0.06, "Missing": 0 },
           "PctHousLess3BR": {"Min": 0, "Max": 1, "Mean": 0.50, "SD": 0.17, "Correl": 0.47, "Median": 0.51, "Mode": 0.53, "Missing": 0 },
           "MedNumBR": {"Min": 0, "Max": 1, "Mean": 0.31, "SD": 0.26, "Correl": -0.36, "Median": 0.5, "Mode": 0.5, "Missing": 0 },
           "HousVacant": {"Min": 0, "Max": 1, "Mean": 0.08, "SD": 0.15, "Correl": 0.42, "Median": 0.03, "Mode": 0.01, "Missing": 0 },
           "PctHousOccup": {"Min": 0, "Max": 1, "Mean": 0.72, "SD": 0.19, "Correl": -0.32, "Median": 0.77, "Mode": 0.88, "Missing": 0 },
           "PctHousOwnOcc": {"Min": 0, "Max": 1, "Mean": 0.55, "SD": 0.19, "Correl": -0.47, "Median": 0.54, "Mode": 0.52, "Missing": 0 },
           "PctVacantBoarded": {"Min": 0, "Max": 1, "Mean": 0.20, "SD": 0.22, "Correl": 0.48, "Median": 0.13, "Mode": 0, "Missing": 0 },
           "PctVacMore6Mos": {"Min": 0, "Max": 1, "Mean": 0.43, "SD": 0.19, "Correl": 0.02, "Median": 0.42, "Mode": 0.44, "Missing": 0 },
           "MedYrHousBuilt": {"Min": 0, "Max": 1, "Mean": 0.49, "SD": 0.23, "Correl": -0.11, "Median": 0.52, "Mode": 0, "Missing": 0 },
           "PctHousNoPhone": {"Min": 0, "Max": 1, "Mean": 0.26, "SD": 0.24, "Correl": 0.49, "Median": 0.185, "Mode": 0.01, "Missing": 0 },
           "PctWOFullPlumb": {"Min": 0, "Max": 1, "Mean": 0.24, "SD": 0.21, "Correl": 0.36, "Median": 0.19, "Mode": 0, "Missing": 0 },
           "OwnOccLowQuart": {"Min": 0, "Max": 1, "Mean": 0.26, "SD": 0.22, "Correl": -0.21, "Median": 0.18, "Mode": 0.09, "Missing": 0 },
           "OwnOccMedVal": {"Min": 0, "Max": 1, "Mean": 0.26, "SD": 0.23, "Correl": -0.19, "Median": 0.17, "Mode": 0.08, "Missing": 0 },
           "OwnOccHiQuart": {"Min": 0, "Max": 1, "Mean": 0.27, "SD": 0.24, "Correl": -0.17, "Median": 0.18, "Mode": 0.08, "Missing": 0 },
           "RentLowQ": {"Min": 0, "Max": 1, "Mean": 0.35, "SD": 0.22, "Correl": -0.25, "Median": 0.31, "Mode": 0.13, "Missing": 0 },
           "RentMedian": {"Min": 0, "Max": 1, "Mean": 0.37, "SD": 0.21, "Correl": -0.24, "Median": 0.33, "Mode": 0.19, "Missing": 0 },
           "RentHighQ": {"Min": 0, "Max": 1, "Mean": 0.42, "SD": 0.25, "Correl": -0.23, "Median": 0.37, "Mode": 1, "Missing": 0 },
           "MedRent": {"Min": 0, "Max": 1, "Mean": 0.38, "SD": 0.21, "Correl": -0.24, "Median": 0.34, "Mode": 0.17, "Missing": 0 },
           "MedRentPctHousInc": {"Min": 0, "Max": 1, "Mean": 0.49, "SD": 0.17, "Correl": 0.33, "Median": 0.48, "Mode": 0.4, "Missing": 0 },
           "MedOwnCostPctInc": {"Min": 0, "Max": 1, "Mean": 0.45, "SD": 0.19, "Correl": 0.06, "Median": 0.45, "Mode": 0.41, "Missing": 0 },
           "MedOwnCostPctIncNoMtg": {"Min": 0, "Max": 1, "Mean": 0.40, "SD": 0.19, "Correl": 0.05, "Median": 0.37, "Mode": 0.24, "Missing": 0 },
           "NumInShelters": {"Min": 0, "Max": 1, "Mean": 0.03, "SD": 0.10, "Correl": 0.38, "Median": 0, "Mode": 0, "Missing": 0 },
           "NumStreet": {"Min": 0, "Max": 1, "Mean": 0.02, "SD": 0.10, "Correl": 0.34, "Median": 0, "Mode": 0, "Missing": 0 },
           "PctForeignBorn": {"Min": 0, "Max": 1, "Mean": 0.22, "SD": 0.23, "Correl": 0.19, "Median": 0.13, "Mode": 0.03, "Missing": 0 },
           "PctBornSameState": {"Min": 0, "Max": 1, "Mean": 0.61, "SD": 0.20, "Correl": -0.08, "Median": 0.63, "Mode": 0.78, "Missing": 0 },
           "PctSameHouse85": {"Min": 0, "Max": 1, "Mean": 0.54, "SD": 0.18, "Correl": -0.16, "Median": 0.54, "Mode": 0.59, "Missing": 0 },
           "PctSameCity85": {"Min": 0, "Max": 1, "Mean": 0.63, "SD": 0.20, "Correl": 0.08, "Median": 0.67, "Mode": 0.74, "Missing": 0 },
           "PctSameState85": {"Min": 0, "Max": 1, "Mean": 0.65, "SD": 0.20, "Correl": -0.02, "Median": 0.7, "Mode": 0.79, "Missing": 0 },
           "LemasSwornFT": {"Min": 0, "Max": 1, "Mean": 0.07, "SD": 0.14, "Correl": 0.34, "Median": 0.02, "Mode": 0.02, "Missing": 1675 },
           "LemasSwFTPerPop": {"Min": 0, "Max": 1, "Mean": 0.22, "SD": 0.16, "Correl": 0.15, "Median": 0.18, "Mode": 0.2, "Missing": 1675 },
           "LemasSwFTFieldOps": {"Min": 0, "Max": 1, "Mean": 0.92, "SD": 0.13, "Correl": -0.33, "Median": 0.97, "Mode": 0.98, "Missing": 1675 },
           "LemasSwFTFieldPerPop": {"Min": 0, "Max": 1, "Mean": 0.25, "SD": 0.16, "Correl": 0.16, "Median": 0.21, "Mode": 0.19, "Missing": 1675 },
           "LemasTotalReq": {"Min": 0, "Max": 1, "Mean": 0.10, "SD": 0.16, "Correl": 0.35, "Median": 0.04, "Mode": 0.02, "Missing": 1675 },
           "LemasTotReqPerPop": {"Min": 0, "Max": 1, "Mean": 0.22, "SD": 0.16, "Correl": 0.27, "Median": 0.17, "Mode": 0.14, "Missing": 1675 },
           "PolicReqPerOffic": {"Min": 0, "Max": 1, "Mean": 0.34, "SD": 0.20, "Correl": 0.17, "Median": 0.29, "Mode": 0.23, "Missing": 1675 },
           "PolicPerPop": {"Min": 0, "Max": 1, "Mean": 0.22, "SD": 0.16, "Correl": 0.15, "Median": 0.18, "Mode": 0.2, "Missing": 1675 },
           "RacialMatchCommPol": {"Min": 0, "Max": 1, "Mean": 0.69, "SD": 0.23, "Correl": -0.46, "Median": 0.74, "Mode": 0.78, "Missing": 1675 },
           "PctPolicWhite": {"Min": 0, "Max": 1, "Mean": 0.73, "SD": 0.22, "Correl": -0.44, "Median": 0.78, "Mode": 0.72, "Missing": 1675 },
           "PctPolicBlack": {"Min": 0, "Max": 1, "Mean": 0.22, "SD": 0.24, "Correl": 0.54, "Median": 0.12, "Mode": 0, "Missing": 1675 },
           "PctPolicHisp": {"Min": 0, "Max": 1, "Mean": 0.13, "SD": 0.20, "Correl": 0.12, "Median": 0.06, "Mode": 0, "Missing": 1675 },
           "PctPolicAsian": {"Min": 0, "Max": 1, "Mean": 0.11, "SD": 0.23, "Correl": 0.10, "Median": 0, "Mode": 0, "Missing": 1675 },
           "PctPolicMinor": {"Min": 0, "Max": 1, "Mean": 0.26, "SD": 0.23, "Correl": 0.49, "Median": 0.2, "Mode": 0.07, "Missing": 1675 },
           "OfficAssgnDrugUnits": {"Min": 0, "Max": 1, "Mean": 0.08, "SD": 0.12, "Correl": 0.34, "Median": 0.04, "Mode": 0.03, "Missing": 1675 },
           "NumKindsDrugsSeiz": {"Min": 0, "Max": 1, "Mean": 0.56, "SD": 0.20, "Correl": 0.13, "Median": 0.57, "Mode": 0.57, "Missing": 1675 },
           "PolicAveOTWorked": {"Min": 0, "Max": 1, "Mean": 0.31, "SD": 0.23, "Correl": 0.03, "Median": 0.26, "Mode": 0.19, "Missing": 1675 },
           "LandArea": {"Min": 0, "Max": 1, "Mean": 0.07, "SD": 0.11, "Correl": 0.20, "Median": 0.04, "Mode": 0.01, "Missing": 0 },
           "PopDens": {"Min": 0, "Max": 1, "Mean": 0.23, "SD": 0.20, "Correl": 0.28, "Median": 0.17, "Mode": 0.09, "Missing": 0 },
           "PctUsePubTrans": {"Min": 0, "Max": 1, "Mean": 0.16, "SD": 0.23, "Correl": 0.15, "Median": 0.07, "Mode": 0.01, "Missing": 0 },
           "PolicCars": {"Min": 0, "Max": 1, "Mean": 0.16, "SD": 0.21, "Correl": 0.38, "Median": 0.08, "Mode": 0.02, "Missing": 1675 },
           "PolicOperBudg": {"Min": 0, "Max": 1, "Mean": 0.08, "SD": 0.14, "Correl": 0.34, "Median": 0.03, "Mode": 0.02, "Missing": 1675 },
           "LemasPctPolicOnPatr": {"Min": 0, "Max": 1, "Mean": 0.70, "SD": 0.21, "Correl": -0.08, "Median": 0.75, "Mode": 0.74, "Missing": 1675 },
           "LemasGangUnitDeploy": {"Min": 0, "Max": 1, "Mean": 0.44, "SD": 0.41, "Correl": 0.12, "Median": 0.5, "Mode": 0, "Missing": 1675 },
           "LemasPctOfficDrugUn": {"Min": 0, "Max": 1, "Mean": 0.09, "SD": 0.24, "Correl": 0.35, "Median": 0, "Mode": 0, "Missing": 0 },
           "PolicBudgPerPop": {"Min": 0, "Max": 1, "Mean": 0.20, "SD": 0.16, "Correl": 0.10, "Median": 0.15, "Mode": 0.12, "Missing": 1675 },
           "ViolentCrimesPerPop": {"Min": 0, "Max": 1, "Mean": 0.24, "SD": 0.23, "Correl": 1.00, "Median": 0.15, "Mode": 0.03, "Missing": 0}}

CrimeHeader = ["state", "county", "community", "communityname", "fold", "population", "householdsize", "racepctblack", "racePctWhite", "racePctAsian", "racePctHisp", "agePct12t21", "agePct12t29", "agePct16t24", "agePct65up", "numbUrban", "pctUrban", "medIncome", "pctWWage", "pctWFarmSelf", "pctWInvInc", "pctWSocSec", "pctWPubAsst", "pctWRetire", "medFamInc", "perCapInc", "whitePerCap", "blackPerCap", "indianPerCap", "AsianPerCap", "OtherPerCap", "HispPerCap", "NumUnderPov", "PctPopUnderPov", "PctLess9thGrade", "PctNotHSGrad", "PctBSorMore", "PctUnemployed", "PctEmploy", "PctEmplManu", "PctEmplProfServ", "PctOccupManu", "PctOccupMgmtProf", "MalePctDivorce", "MalePctNevMarr", "FemalePctDiv", "TotalPctDiv", "PersPerFam", "PctFam2Par", "PctKids2Par", "PctYoungKids2Par", "PctTeen2Par", "PctWorkMomYoungKids", "PctWorkMom", "NumIlleg", "PctIlleg", "NumImmig", "PctImmigRecent", "PctImmigRec5", "PctImmigRec8", "PctImmigRec10", "PctRecentImmig", "PctRecImmig5", "PctRecImmig8", "PctRecImmig10", "PctSpeakEnglOnly", "PctNotSpeakEnglWell", "PctLargHouseFam", "PctLargHouseOccup", "PersPerOccupHous", "PersPerOwnOccHous", "PersPerRentOccHous", "PctPersOwnOccup", "PctPersDenseHous", "PctHousLess3BR", "MedNumBR", "HousVacant", "PctHousOccup", "PctHousOwnOcc", "PctVacantBoarded", "PctVacMore6Mos", "MedYrHousBuilt", "PctHousNoPhone", "PctWOFullPlumb", "OwnOccLowQuart", "OwnOccMedVal", "OwnOccHiQuart", "RentLowQ", "RentMedian", "RentHighQ", "MedRent", "MedRentPctHousInc", "MedOwnCostPctInc", "MedOwnCostPctIncNoMtg", "NumInShelters", "NumStreet", "PctForeignBorn", "PctBornSameState", "PctSameHouse85", "PctSameCity85", "PctSameState85", "LemasSwornFT", "LemasSwFTPerPop", "LemasSwFTFieldOps", "LemasSwFTFieldPerPop", "LemasTotalReq", "LemasTotReqPerPop", "PolicReqPerOffic", "PolicPerPop", "RacialMatchCommPol", "PctPolicWhite", "PctPolicBlack", "PctPolicHisp", "PctPolicAsian", "PctPolicMinor", "OfficAssgnDrugUnits", "NumKindsDrugsSeiz", "PolicAveOTWorked", "LandArea", "PopDens", "PctUsePubTrans", "PolicCars", "PolicOperBudg", "LemasPctPolicOnPatr", "LemasGangUnitDeploy", "LemasPctOfficDrugUn", "PolicBudgPerPop", "ViolentCrimesPerPop" ]


CrimeHeaderDataOnly = ["population", "householdsize", "racepctblack", "racePctWhite", "racePctAsian", "racePctHisp", "agePct12t21", "agePct12t29", "agePct16t24", "agePct65up", "numbUrban", "pctUrban", "medIncome", "pctWWage", "pctWFarmSelf", "pctWInvInc", "pctWSocSec", "pctWPubAsst", "pctWRetire", "medFamInc", "perCapInc", "whitePerCap", "blackPerCap", "indianPerCap", "AsianPerCap", "OtherPerCap", "HispPerCap", "NumUnderPov", "PctPopUnderPov", "PctLess9thGrade", "PctNotHSGrad", "PctBSorMore", "PctUnemployed", "PctEmploy", "PctEmplManu", "PctEmplProfServ", "PctOccupManu", "PctOccupMgmtProf", "MalePctDivorce", "MalePctNevMarr", "FemalePctDiv", "TotalPctDiv", "PersPerFam", "PctFam2Par", "PctKids2Par", "PctYoungKids2Par", "PctTeen2Par", "PctWorkMomYoungKids", "PctWorkMom", "NumIlleg", "PctIlleg", "NumImmig", "PctImmigRecent", "PctImmigRec5", "PctImmigRec8", "PctImmigRec10", "PctRecentImmig", "PctRecImmig5", "PctRecImmig8", "PctRecImmig10", "PctSpeakEnglOnly", "PctNotSpeakEnglWell", "PctLargHouseFam", "PctLargHouseOccup", "PersPerOccupHous", "PersPerOwnOccHous", "PersPerRentOccHous", "PctPersOwnOccup", "PctPersDenseHous", "PctHousLess3BR", "MedNumBR", "HousVacant", "PctHousOccup", "PctHousOwnOcc", "PctVacantBoarded", "PctVacMore6Mos", "MedYrHousBuilt", "PctHousNoPhone", "PctWOFullPlumb", "OwnOccLowQuart", "OwnOccMedVal", "OwnOccHiQuart", "RentLowQ", "RentMedian", "RentHighQ", "MedRent", "MedRentPctHousInc", "MedOwnCostPctInc", "MedOwnCostPctIncNoMtg", "NumInShelters", "NumStreet", "PctForeignBorn", "PctBornSameState", "PctSameHouse85", "PctSameCity85", "PctSameState85", "LemasSwornFT", "LemasSwFTPerPop", "LemasSwFTFieldOps", "LemasSwFTFieldPerPop", "LemasTotalReq", "LemasTotReqPerPop", "PolicReqPerOffic", "PolicPerPop", "RacialMatchCommPol", "PctPolicWhite", "PctPolicBlack", "PctPolicHisp", "PctPolicAsian", "PctPolicMinor", "OfficAssgnDrugUnits", "NumKindsDrugsSeiz", "PolicAveOTWorked", "LandArea", "PopDens", "PctUsePubTrans", "PolicCars", "PolicOperBudg", "LemasPctPolicOnPatr", "LemasGangUnitDeploy", "LemasPctOfficDrugUn", "PolicBudgPerPop"]

CrimeHeaderSansBadData = ["population", "householdsize", "racepctblack", "racePctWhite", "racePctAsian", "racePctHisp", "agePct12t21", "agePct12t29", "agePct16t24", "agePct65up", "numbUrban", "pctUrban", "medIncome", "pctWWage", "pctWFarmSelf", "pctWInvInc", "pctWSocSec", "pctWPubAsst", "pctWRetire", "medFamInc", "perCapInc", "whitePerCap", "blackPerCap", "indianPerCap", "AsianPerCap", "OtherPerCap", "HispPerCap", "NumUnderPov", "PctPopUnderPov", "PctLess9thGrade", "PctNotHSGrad", "PctBSorMore", "PctUnemployed", "PctEmploy", "PctEmplManu", "PctEmplProfServ", "PctOccupManu", "PctOccupMgmtProf", "MalePctDivorce", "MalePctNevMarr", "FemalePctDiv", "TotalPctDiv", "PersPerFam", "PctFam2Par", "PctKids2Par", "PctYoungKids2Par", "PctTeen2Par", "PctWorkMomYoungKids", "PctWorkMom", "NumIlleg", "PctIlleg", "NumImmig", "PctImmigRecent", "PctImmigRec5", "PctImmigRec8", "PctImmigRec10", "PctRecentImmig", "PctRecImmig5", "PctRecImmig8", "PctRecImmig10", "PctSpeakEnglOnly", "PctNotSpeakEnglWell", "PctLargHouseFam", "PctLargHouseOccup", "PersPerOccupHous", "PersPerOwnOccHous", "PersPerRentOccHous", "PctPersOwnOccup", "PctPersDenseHous", "PctHousLess3BR", "MedNumBR", "HousVacant", "PctHousOccup", "PctHousOwnOcc", "PctVacantBoarded", "PctVacMore6Mos", "MedYrHousBuilt", "PctHousNoPhone", "PctWOFullPlumb", "OwnOccLowQuart", "OwnOccMedVal", "OwnOccHiQuart", "RentLowQ", "RentMedian", "RentHighQ", "MedRent", "MedRentPctHousInc", "MedOwnCostPctInc", "MedOwnCostPctIncNoMtg", "NumInShelters", "NumStreet", "PctForeignBorn", "PctBornSameState", "PctSameHouse85", "PctSameCity85", "PctSameState85", "LandArea", "PopDens", "PctUsePubTrans", "LemasPctOfficDrugUn"]

CrimeHeaderNoRace = ["population", "householdsize", "agePct12t21", "agePct12t29", "agePct16t24", "agePct65up", "numbUrban", "pctUrban", "medIncome", "pctWWage", "pctWFarmSelf", "pctWInvInc", "pctWSocSec", "pctWPubAsst", "pctWRetire", "medFamInc", "perCapInc", "whitePerCap", "blackPerCap", "indianPerCap", "AsianPerCap", "OtherPerCap", "HispPerCap", "NumUnderPov", "PctPopUnderPov", "PctLess9thGrade", "PctNotHSGrad", "PctBSorMore", "PctUnemployed", "PctEmploy", "PctEmplManu", "PctEmplProfServ", "PctOccupManu", "PctOccupMgmtProf", "MalePctDivorce", "MalePctNevMarr", "FemalePctDiv", "TotalPctDiv", "PersPerFam", "PctFam2Par", "PctKids2Par", "PctYoungKids2Par", "PctTeen2Par", "PctWorkMomYoungKids", "PctWorkMom", "NumIlleg", "PctIlleg", "NumImmig", "PctImmigRecent", "PctImmigRec5", "PctImmigRec8", "PctImmigRec10", "PctRecentImmig", "PctRecImmig5", "PctRecImmig8", "PctRecImmig10", "PctSpeakEnglOnly", "PctNotSpeakEnglWell", "PctLargHouseFam", "PctLargHouseOccup", "PersPerOccupHous", "PersPerOwnOccHous", "PersPerRentOccHous", "PctPersOwnOccup", "PctPersDenseHous", "PctHousLess3BR", "MedNumBR", "HousVacant", "PctHousOccup", "PctHousOwnOcc", "PctVacantBoarded", "PctVacMore6Mos", "MedYrHousBuilt", "PctHousNoPhone", "PctWOFullPlumb", "OwnOccLowQuart", "OwnOccMedVal", "OwnOccHiQuart", "RentLowQ", "RentMedian", "RentHighQ", "MedRent", "MedRentPctHousInc", "MedOwnCostPctInc", "MedOwnCostPctIncNoMtg", "NumInShelters", "NumStreet", "PctForeignBorn", "PctBornSameState", "PctSameHouse85", "PctSameCity85", "PctSameState85", "LandArea", "PopDens", "PctUsePubTrans", "LemasPctOfficDrugUn"]
