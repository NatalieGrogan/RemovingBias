import sklearn as scikit
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import linear_model
import data

UCICorrelData={}
for k,d in data.summary.items():
    UCICorrelData[k] = d["Correl"]

CrimeMatrix = pd.read_csv(filepath_or_buffer = 'communities.data', header = None, names = data.CrimeHeader, na_values="?")

#Removing the Non-Numerical data and seperating the data we want to make predictions on
TrimmedCrimeMatrix =pd.DataFrame(CrimeMatrix[data.CrimeHeaderSansBadData])
CopyTrimmedCrimeMatrix = TrimmedCrimeMatrix.copy()
CopyCopyCrimeMatrix = TrimmedCrimeMatrix.copy()
CopyCopyCopyCrimeMatrix = TrimmedCrimeMatrix.copy()
TargetData = CrimeMatrix['ViolentCrimesPerPop'].copy()

#Creating the LinearModel Tools
regrMean = linear_model.LinearRegression(fit_intercept=True)

#Need to fill in the missing data so that we can run PCA/Other analysis
#Theres 3 different standard Imputers; Mean, Median, Mode
#Choose to try out Mean and Median to see if there was a large difference
#aka does the method we used to Impute matter?
#Theres also an Experimental Imputer that seems smarter but chose not to use it b/c it's experimental.
impMean = SimpleImputer(missing_values =np.nan, strategy = 'mean')
meanCrimeMatrix = pd.DataFrame(impMean.fit_transform(TrimmedCrimeMatrix),columns=data.CrimeHeaderSansBadData)

#Rounding to bring back in line with original Signifigant Digits
meanCrimeMatrix = meanCrimeMatrix.round(2)

meanCrimeMatrixNoRace, raceArray = data.modifyRace(meanCrimeMatrix,data.CrimeHeaderNoRace)

#normalizing %WhiteToNonWht
#maybe isn't good idea but I'm doing it anyway
maxRace = np.amax(raceArray)
for i in range(0,raceArray.shape[0]):
    raceArray[i] = raceArray[i] / maxRace

raceArray = pd.DataFrame(raceArray)

#creating matrix that includes race 
meanCrimeMatrixWithRace = meanCrimeMatrixNoRace.copy()
meanCrimeMatrixWithRace.insert(meanCrimeMatrixNoRace.shape[1], 'pctWhtToNonWht', raceArray)

#creating matrix for storing data for analysis, column 1 and 2 are which cities I'm comparing, others will be for storing calculations midway
#this matrix is for most similar cities
predictions = np.empty(shape=(9,9), dtype=float)
for i in range(0,predictions.shape[0]):
    predictions[i,0] = data.mostCloseCities[i][0]
    predictions[i,1] = data.mostCloseCities[i][1]
    predictions[i,2] = 0
    predictions[i,3] = 0
    predictions[i,2] = 0
    predictions[i,3] = 0
    predictions[i,4] = 0
    predictions[i,5] = 0
    predictions[i,6] = 0
    predictions[i,7] = 0
    predictions[i,8] = raceArray.iloc[data.mostCloseCities[i][0]] - raceArray.iloc[data.mostCloseCities[i][1]]

#creating 2nd matrix for storing data for analysis, column 1 and 2 are which cities I'm comparing, others will be for storing calculations midway
#this matrix is for most different cities
predictions2 = np.empty(shape=(10,9), dtype=float)
for i in range(0,predictions2.shape[0]):
    predictions2[i,0] = data.mostDifCities[i][0]
    predictions2[i,1] = data.mostDifCities[i][1]
    predictions2[i,2] = 0
    predictions2[i,3] = 0
    predictions2[i,2] = 0
    predictions2[i,3] = 0
    predictions2[i,4] = 0
    predictions2[i,5] = 0
    predictions2[i,6] = 0
    predictions2[i,7] = 0
    predictions2[i,8] = raceArray.iloc[data.mostDifCities[i][0]] - raceArray.iloc[data.mostDifCities[i][1]]

#Calc LinReg w/ Race
regrMean.fit(meanCrimeMatrixWithRace, TargetData)
LinRegParamWithRace = regrMean.coef_
PredictionsWithRace = regrMean.predict(meanCrimeMatrixWithRace)

#Doing analysis to find what the control algo predicts violent crime will be in the two cities and find the difference
for i in range(0,predictions2.shape[0]):
    city3 = int(predictions2[i,0])
    city4 = int(predictions2[i,1])
    if i == 9:
        for k in range(0,LinRegParamWithRace.shape[0]):
            param = LinRegParamWithRace[k]
            predictions2[i,2] += meanCrimeMatrixWithRace.iloc[city3,k] * param
            predictions2[i,3] += meanCrimeMatrixWithRace.iloc[city4,k] * param
    else:
        city1 = int(predictions[i,0])
        city2 = int(predictions[i,1])
        for k in range(0,LinRegParamWithRace.shape[0]):
            param = LinRegParamWithRace[k]
            predictions[i,2] += meanCrimeMatrixWithRace.iloc[city1,k] * param
            predictions[i,3] += meanCrimeMatrixWithRace.iloc[city2,k] * param
            predictions2[i,2] += meanCrimeMatrixWithRace.iloc[city3,k] * param
            predictions2[i,3] += meanCrimeMatrixWithRace.iloc[city4,k] * param
        predictions[i,4] = predictions[i,2] - predictions[i,3]
    predictions2[i,4] = predictions2[i,2] - predictions2[i,3]

#Calc X-Tilde and then LinReg w/o Race
X_Tilde = data.calcLinearReg(meanCrimeMatrixNoRace,raceArray,True)
regrMean.fit(X_Tilde, TargetData)
LinRegParamNoRace = regrMean.coef_
PredictionsNoRace = regrMean.predict(meanCrimeMatrixNoRace)

#Doing analysis to find what the modified algo predicts violent crime will be in the two cities and find the difference
for i in range(0,predictions2.shape[0]):
    city3 = int(predictions2[i,0])
    city4 = int(predictions2[i,1])
    if i == 9:
        for k in range(0,LinRegParamNoRace.shape[0]):
            param = LinRegParamNoRace[k]
            predictions2[i,5] += meanCrimeMatrixNoRace.iloc[city3,k] * param
            predictions2[i,6] += meanCrimeMatrixNoRace.iloc[city4,k] * param
    else:
        city1 = int(predictions[i,0])
        city2 = int(predictions[i,1])
        for k in range(0,LinRegParamNoRace.shape[0]):
            param = LinRegParamNoRace[k]
            predictions[i,5] += meanCrimeMatrixNoRace.iloc[city1,k] * param
            predictions[i,6] += meanCrimeMatrixNoRace.iloc[city2,k] * param
            predictions2[i,5] += meanCrimeMatrixNoRace.iloc[city3,k] * param
            predictions2[i,6] += meanCrimeMatrixNoRace.iloc[city4,k] * param
        predictions[i,7] = predictions[i,5] - predictions[i,6]
    predictions2[i,7] = predictions2[i,5] - predictions2[i,6]

#Need to use adjusted R-Squared since we have a large number of explanatory variables
# doesn't super change the results, I hope
AdjustedRSquaredWithRace = 1- (1-r2_score(TargetData, PredictionsWithRace))*(1993/(1993-97))
AdjustedRSquaredNoRace = 1- (1-r2_score(TargetData, PredictionsNoRace))*(1993/(1993-96))

print('Coefficient of determination w/o Race: %.4f' % AdjustedRSquaredNoRace)
print('Coefficient of determination w/ Race: %.4f \n' % AdjustedRSquaredWithRace)
#Printing the results of Analysis
print('The values on the WR & NR lines are difference between the predicted')
print('   violent crime per pop stats for each city, smaller absolute values are better')
print('WR - With Race,   NR - No Race')
print('The Race line is the difference between the two cities %WhtTo%PoC metric\n')
print('Most Similar cities:')
print('Cities: 848-983  595-1893 389-1111 1033-1224 27-1473 668-699 1658-1926 1461-1768 1171-1413')
print('WR  : ',np.array2string(predictions[:9,3].round(5)))
print('NR  : ',np.array2string(predictions[:9,7].round(5)))
print('Race: ',np.array2string(predictions[:9,8].round(5)))
print('         *The city numbers are the index number of the two cities being compared \n')
print('Most Dissimilar Cities:')
print('Cities: 1134-1615 139-1134 910-1134 849-1134 1158-1847 287-1134 657-1134 744-1134 492-1134 1500-1847')
print('WR  : ',np.array2string(predictions2[:10,3].round(5)))
print('NR  : ',np.array2string(predictions2[:10,7].round(5)))
print('Race: ',np.array2string(predictions2[:10,8].round(5)))
print('         *The city numbers are the index number of the two cities being compared \n')



correctDimOneMatrix = np.empty(shape=(2,LinRegParamWithRace.shape[0]), dtype=float)

for i in range(0,LinRegParamWithRace.shape[0]):     
    if i == LinRegParamWithRace.shape[0]-1:
        correctDimOneMatrix[0,i] = LinRegParamWithRace[i]
    else:
        correctDimOneMatrix[1,i] = LinRegParamNoRace[i]
        correctDimOneMatrix[0,i] = LinRegParamWithRace[i]

data.CrimeHeaderNoRace.append('pctWhtToNonWht')
LinRegParams= pd.DataFrame(correctDimOneMatrix, columns=data.CrimeHeaderNoRace)
LinRegParams.round(3).T.to_string(buf='Results')


#This commented out section is what I used to find the top 10 most similar and dissimilar cities, excluding race
#Note this is very slow because its calculating the pairwise difference between each ciy and every other city - aka ~1/2 of a 1994x1994 matrix
#I suggest leaving it commented out
#Also note there is a bug in this code for the first city found when looking for most similar cities.
#I choose not to find it because the other 9 cities seemed to be correct - their metrics were very similar
'''
differenceMatrix= np.empty(shape=(meanCrimeMatrixWithRace.shape[0],meanCrimeMatrixWithRace.shape[0]), dtype=float)



#calculates the sum of the absolute value of the difference in all dimensions except race between two cities i and k.
for i in range(0,meanCrimeMatrixNoRace.shape[0]):
    for k in range(i+1,meanCrimeMatrixNoRace.shape[0]):
        standInArray = abs(meanCrimeMatrixNoRace.iloc[i] - meanCrimeMatrixNoRace.iloc[k])
        differenceMatrix[i,k] = round(standInArray.sum(),4)

#replaces the diagonal and the mirror half of the matrix with NaN to prevent confusion in future calculations.
for i in range(0,meanCrimeMatrixNoRace.shape[0]):
    for k in range(0,i):
        differenceMatrix[i,k] = np.nan
    differenceMatrix[i,i] = np.nan

minDifCity = np.empty(shape=(1994,3), dtype=float)
#need to find the 10 min value in the difference matrix
for i in range(0,meanCrimeMatrixNoRace.shape[0]):
    for k in range(i+1,meanCrimeMatrixNoRace.shape[0]):
        if k == i+1:
            minDifCity[i,0] = differenceMatrix[i,k]
            minDifCity[i,1] = i
            minDifCity[i,2] = k
        else:
            if differenceMatrix[i,k] > minDifCity[i,0]:
               minDifCity[i,0] = differenceMatrix[i,k]
               minDifCity[i,1] = i
               minDifCity[i,2] = k

#Note you must also change the above loop else if statement to be either >, for most different, or <, for most similar, cities
#This loop will find the most similar cities when you use np.argmin AND set the minDifCity to 10,000 after storing it
#This loop will find the most similar cities when you use np.argmax AND set the minDifCity to -10,000 after storing it
for i in range(0,10):
    value = np.argmax(minDifCity,axis=0)
    print(minDifCity[value[0],1],minDifCity[value[0],2])
    minDifCity[value[0],0] = -10000
    print(value)
'''    


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
'''
#Attribute Information:

#Attribute Information: (122 predictive, 5 non-predictive, 1 goal) 
#-- state: US state (by number) - not counted as predictive above, but if considered, should be consided nominal (nominal) 
#-- county: numeric code for county - not predictive, and many missing values (numeric) 
#-- community: numeric code for community - not predictive and many missing values (numeric) 
#-- communityname: community name - not predictive - for information only (string) 
#-- fold: fold number for non-random 10 fold cross validation, potentially useful for debugging, paired tests - not predictive (numeric) 
#-- population: population for community: (numeric - decimal) 
#-- householdsize: mean people per household (numeric - decimal) 
#-- racepctblack: percentage of population that is african american (numeric - decimal) 
#-- racePctWhite: percentage of population that is caucasian (numeric - decimal) 
#-- racePctAsian: percentage of population that is of asian heritage (numeric - decimal) 
#-- racePctHisp: percentage of population that is of hispanic heritage (numeric - decimal) 
#-- agePct12t21: percentage of population that is 12-21 in age (numeric - decimal) 
#-- agePct12t29: percentage of population that is 12-29 in age (numeric - decimal) 
#-- agePct16t24: percentage of population that is 16-24 in age (numeric - decimal) 
#-- agePct65up: percentage of population that is 65 and over in age (numeric - decimal) 
#-- numbUrban: number of people living in areas classified as urban (numeric - decimal) 
#-- pctUrban: percentage of people living in areas classified as urban (numeric - decimal) 
#-- medIncome: median household income (numeric - decimal) 
#-- pctWWage: percentage of households with wage or salary income in 1989 (numeric - decimal) 
#-- pctWFarmSelf: percentage of households with farm or self employment income in 1989 (numeric - decimal) 
#-- pctWInvInc: percentage of households with investment / rent income in 1989 (numeric - decimal) 
#-- pctWSocSec: percentage of households with social security income in 1989 (numeric - decimal) 
#-- pctWPubAsst: percentage of households with public assistance income in 1989 (numeric - decimal) 
#-- pctWRetire: percentage of households with retirement income in 1989 (numeric - decimal) 
#-- medFamInc: median family income (differs from household income for non-family households) (numeric - decimal) 
#-- perCapInc: per capita income (numeric - decimal) 
#-- whitePerCap: per capita income for caucasians (numeric - decimal) 
#-- blackPerCap: per capita income for african americans (numeric - decimal) 
#-- indianPerCap: per capita income for native americans (numeric - decimal) 
#-- AsianPerCap: per capita income for people with asian heritage (numeric - decimal) 
#-- OtherPerCap: per capita income for people with 'other' heritage (numeric - decimal) 
#-- HispPerCap: per capita income for people with hispanic heritage (numeric - decimal) 
#-- NumUnderPov: number of people under the poverty level (numeric - decimal) 
#-- PctPopUnderPov: percentage of people under the poverty level (numeric - decimal) 
#-- PctLess9thGrade: percentage of people 25 and over with less than a 9th grade education (numeric - decimal) 
#-- PctNotHSGrad: percentage of people 25 and over that are not high school graduates (numeric - decimal) 
#-- PctBSorMore: percentage of people 25 and over with a bachelors degree or higher education (numeric - decimal) 
#-- PctUnemployed: percentage of people 16 and over, in the labor force, and unemployed (numeric - decimal) 
#-- PctEmploy: percentage of people 16 and over who are employed (numeric - decimal) 
#-- PctEmplManu: percentage of people 16 and over who are employed in manufacturing (numeric - decimal) 
#-- PctEmplProfServ: percentage of people 16 and over who are employed in professional services (numeric - decimal) 
#-- PctOccupManu: percentage of people 16 and over who are employed in manufacturing (numeric - decimal) ######## 
#-- PctOccupMgmtProf: percentage of people 16 and over who are employed in management or professional occupations (numeric - decimal) 
#-- MalePctDivorce: percentage of males who are divorced (numeric - decimal) 
#-- MalePctNevMarr: percentage of males who have never married (numeric - decimal) 
#-- FemalePctDiv: percentage of females who are divorced (numeric - decimal) 
#-- TotalPctDiv: percentage of population who are divorced (numeric - decimal) 
#-- PersPerFam: mean number of people per family (numeric - decimal) 
#-- PctFam2Par: percentage of families (with kids) that are headed by two parents (numeric - decimal) 
#-- PctKids2Par: percentage of kids in family housing with two parents (numeric - decimal) 
#-- PctYoungKids2Par: percent of kids 4 and under in two parent households (numeric - decimal) 
#-- PctTeen2Par: percent of kids age 12-17 in two parent households (numeric - decimal) 
#-- PctWorkMomYoungKids: percentage of moms of kids 6 and under in labor force (numeric - decimal) 
#-- PctWorkMom: percentage of moms of kids under 18 in labor force (numeric - decimal) 
#-- NumIlleg: number of kids born to never married (numeric - decimal) 
#-- PctIlleg: percentage of kids born to never married (numeric - decimal) 
#-- NumImmig: total number of people known to be foreign born (numeric - decimal) 
#-- PctImmigRecent: percentage of _immigrants_ who immigated within last 3 years (numeric - decimal) 
#-- PctImmigRec5: percentage of _immigrants_ who immigated within last 5 years (numeric - decimal) 
#-- PctImmigRec8: percentage of _immigrants_ who immigated within last 8 years (numeric - decimal) 
#-- PctImmigRec10: percentage of _immigrants_ who immigated within last 10 years (numeric - decimal) 
#-- PctRecentImmig: percent of _population_ who have immigrated within the last 3 years (numeric - decimal) 
#-- PctRecImmig5: percent of _population_ who have immigrated within the last 5 years (numeric - decimal) 
#-- PctRecImmig8: percent of _population_ who have immigrated within the last 8 years (numeric - decimal) 
#-- PctRecImmig10: percent of _population_ who have immigrated within the last 10 years (numeric - decimal) 
#-- PctSpeakEnglOnly: percent of people who speak only English (numeric - decimal) 
#-- PctNotSpeakEnglWell: percent of people who do not speak English well (numeric - decimal) 
#-- PctLargHouseFam: percent of family households that are large (6 or more) (numeric - decimal) 
#-- PctLargHouseOccup: percent of all occupied households that are large (6 or more people) (numeric - decimal) 
#-- PersPerOccupHous: mean persons per household (numeric - decimal) 
#-- PersPerOwnOccHous: mean persons per owner occupied household (numeric - decimal) 
#-- PersPerRentOccHous: mean persons per rental household (numeric - decimal) 
#-- PctPersOwnOccup: percent of people in owner occupied households (numeric - decimal) 
#-- PctPersDenseHous: percent of persons in dense housing (more than 1 person per room) (numeric - decimal) 
#-- PctHousLess3BR: percent of housing units with less than 3 bedrooms (numeric - decimal) 
#-- MedNumBR: median number of bedrooms (numeric - decimal) 
#-- HousVacant: number of vacant households (numeric - decimal) 
#-- PctHousOccup: percent of housing occupied (numeric - decimal) 
#-- PctHousOwnOcc: percent of households owner occupied (numeric - decimal) 
#-- PctVacantBoarded: percent of vacant housing that is boarded up (numeric - decimal) 
#-- PctVacMore6Mos: percent of vacant housing that has been vacant more than 6 months (numeric - decimal) 
#-- MedYrHousBuilt: median year housing units built (numeric - decimal) 
#-- PctHousNoPhone: percent of occupied housing units without phone (in 1990, this was rare!) (numeric - decimal) 
#-- PctWOFullPlumb: percent of housing without complete plumbing facilities (numeric - decimal) 
#-- OwnOccLowQuart: owner occupied housing - lower quartile value (numeric - decimal) 
#-- OwnOccMedVal: owner occupied housing - median value (numeric - decimal) 
#-- OwnOccHiQuart: owner occupied housing - upper quartile value (numeric - decimal) 
#-- RentLowQ: rental housing - lower quartile rent (numeric - decimal) 
#-- RentMedian: rental housing - median rent (Census variable H32B from file STF1A) (numeric - decimal) 
#-- RentHighQ: rental housing - upper quartile rent (numeric - decimal) 
#-- MedRent: median gross rent (Census variable H43A from file STF3A - includes utilities) (numeric - decimal) 
#-- MedRentPctHousInc: median gross rent as a percentage of household income (numeric - decimal) 
#-- MedOwnCostPctInc: median owners cost as a percentage of household income - for owners with a mortgage (numeric - decimal) 
#-- MedOwnCostPctIncNoMtg: median owners cost as a percentage of household income - for owners without a mortgage (numeric - decimal) 
#-- NumInShelters: number of people in homeless shelters (numeric - decimal) 
#-- NumStreet: number of homeless people counted in the street (numeric - decimal) 
#-- PctForeignBorn: percent of people foreign born (numeric - decimal) 
#-- PctBornSameState: percent of people born in the same state as currently living (numeric - decimal) 
#-- PctSameHouse85: percent of people living in the same house as in 1985 (5 years before) (numeric - decimal) 
#-- PctSameCity85: percent of people living in the same city as in 1985 (5 years before) (numeric - decimal) 
#-- PctSameState85: percent of people living in the same state as in 1985 (5 years before) (numeric - decimal) 
#-- LemasSwornFT: number of sworn full time police officers (numeric - decimal) 
#-- LemasSwFTPerPop: sworn full time police officers per 100K population (numeric - decimal) 
#-- LemasSwFTFieldOps: number of sworn full time police officers in field operations (on the street as opposed to administrative etc) (numeric - dec#imal) 
#-- LemasSwFTFieldPerPop: sworn full time police officers in field operations (on the street as opposed to administrative etc) per 100K population (#numeric - decimal) 
#-- LemasTotalReq: total requests for police (numeric - decimal) 
#-- LemasTotReqPerPop: total requests for police per 100K popuation (numeric - decimal) 
#-- PolicReqPerOffic: total requests for police per police officer (numeric - decimal) 
#-- PolicPerPop: police officers per 100K population (numeric - decimal) 
#-- RacialMatchCommPol: a measure of the racial match between the community and the police force. High values indicate proportions in community and #police force are similar (numeric - decimal) 
#-- PctPolicWhite: percent of police that are caucasian (numeric - decimal) 
#-- PctPolicBlack: percent of police that are african american (numeric - decimal) 
#-- PctPolicHisp: percent of police that are hispanic (numeric - decimal) 
#-- PctPolicAsian: percent of police that are asian (numeric - decimal) 
#-- PctPolicMinor: percent of police that are minority of any kind (numeric - decimal) 
#-- OfficAssgnDrugUnits: number of officers assigned to special drug units (numeric - decimal) 
#-- NumKindsDrugsSeiz: number of different kinds of drugs seized (numeric - decimal) 
#-- PolicAveOTWorked: police average overtime worked (numeric - decimal) 
#-- LandArea: land area in square miles (numeric - decimal) 
#-- PopDens: population density in persons per square mile (numeric - decimal) 
#-- PctUsePubTrans: percent of people using public transit for commuting (numeric - decimal) 
#-- PolicCars: number of police cars (numeric - decimal) 
#-- PolicOperBudg: police operating budget (numeric - decimal) 
#-- LemasPctPolicOnPatr: percent of sworn full time police officers on patrol (numeric - decimal) 
#-- LemasGangUnitDeploy: gang unit deployed (numeric - decimal - but really ordinal - 0 means NO, 1 means YES, 0.5 means Part Time) 
#-- LemasPctOfficDrugUn: percent of officers assigned to drug units (numeric - decimal) 
#-- PolicBudgPerPop: police operating budget per population (numeric - decimal) 
#-- ViolentCrimesPerPop: total number of violent crimes per 100K popuation (numeric - decimal) GOAL attribute (to be predicted) 
