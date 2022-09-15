# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 16:22:24 2022

@author: User
"""
from dataSourcing import irishCarPriceDatabase
import yaml


from pathlib import Path
from joblib import dump
import json

import pandas as pd
from math import log
from datetime import date, datetime
from itertools import combinations
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA


from sklearn.linear_model import Ridge, Lasso, ElasticNet, HuberRegressor, RANSACRegressor, TheilSenRegressor
from sklearn.model_selection import GridSearchCV
from numpy import random




CFG = {
       'csv': True,
       'inputCsv': 'cleandata.csv',
       'outputPath': 'Models',
       'notes': 'HuberRegressor: worst models',
       'valuationDate': '2022-09-07',
       'sampleFeatures': True,
#       'numFeatures': ['age', 'mileage', 'engine', 'mpg', 'owners', 'nct_remaining'],
       'numFeatures': ['age', 'mileage', 'engine', 'mpg', 'owners', 'nct_remaining', 'tax'],
       'catFeatures': ['transmission', 'body', 'color', 'fuel'],
       #'catFeatures': ['transmission', 'body', 'fuel','color'],
       'target': 'price',
       'annual_mileage': 17000,
       'annual_mileage_cutoff': 1000,
       
       'fillna_method': 'median',
       
       
       'test_size': 0.3,
       
       #OneHot Encoder Parameters
       'enc_drop': 'first',
       'enc_sparse': False,
       'enc_handle_unknown': 'ignore',
       
       #PCA Parameters
       'n_components': 0.99,
       
       #Threshold for low variance features
       'variance_threshold': 0,
       
       #GridSearchCV Parameters
       'estimator': 'HuberRegressor',
       'param_grid': {'alpha': [1],
                      'epsilon': [1.35],
                       'max_iter': [600]},
       'scoring': 'neg_mean_squared_error',
       'refit': True,
       'cv': 5,
       'verbose':0,
       'return_train_score': True,
       'output_coef': False,
       'seed': 1023,
       'readHyperParams': None,
       #'readHyperParams': 'alpha.txt',
       'toTrain': 'toTrain.txt'
       }


def fillna(carData, CFG):
    """
    fill NA values in carData before data processing
    """
    medianModelData = carData.groupby(by='model')[CFG['numFeatures']].median()
    medianMakeData = carData.groupby(by='make')[CFG['numFeatures']].median()
    medianData = carData[CFG['numFeatures']].median()
    
    carData=carData.merge(medianModelData[CFG['numFeatures']], how='left', left_on='make', right_index=True)
    for feature in CFG['numFeatures']:
        carData[feature] =  carData[feature+'_x'].fillna(carData[feature+'_y'])
        carData.drop(labels = [feature+'_x', feature+'_y'], axis=1, inplace=True)
        
    carData=carData.merge(medianMakeData[CFG['numFeatures']], how='left', left_on='make', right_index=True)
    for feature in CFG['numFeatures']:
        carData[feature] =  carData[feature+'_x'].fillna(carData[feature+'_y'])
        carData.drop(labels = [feature+'_x', feature+'_y'], axis=1, inplace=True)    
        
    carData.fillna(medianData, inplace=True)
    
    return carData


def getMakeData(X, Y, make, carModel, numFeatures, catFeatures):
    make_X= X[(X['model']==carModel)  & (X['make']==make)]
    make_X = make_X[numFeatures+catFeatures]
    make_Y = Y[make_X.index]
    return make_X, make_Y


def getDistinctMakes(carData):
    makes=[]
    for make in carData[['make', 'model']].values:
        if tuple(make) not in makes:
            makes.append(tuple(make))
    return makes
        
    
def scaleEncodeTrainData(trainData, scalar, enc, CFG):
    
    #Normalize numderical features
    trainData[CFG['numFeatures']] = scalar.fit_transform(trainData[CFG['numFeatures']])
    
    #Encode categorical features
    encodedData = enc.fit_transform(trainData[CFG['catFeatures']])
    encodedColumns= ['encoded_'+str(x) for x in range(len(encodedData[0]))]
    encodedData = pd.DataFrame(encodedData, index=trainData.index, columns = encodedColumns)
    trainData = pd.concat([trainData, encodedData], axis=1)
    trainData.drop(columns=CFG['catFeatures'], inplace=True)#
    
    return trainData, scalar, enc


def scaleEncodeTestData(testData, scalar, enc, CFG):
    
    #Normalize numderical features
    testData[CFG['numFeatures']] = scalar.transform(testData[CFG['numFeatures']])
    
    #Encode categorical features
    encodedData = enc.transform(testData[CFG['catFeatures']])
    encodedColumns= ['encoded_'+str(x) for x in range(len(encodedData[0]))]
    encodedData = pd.DataFrame(encodedData, index=testData.index, columns = encodedColumns)
    testData = pd.concat([testData, encodedData], axis=1)
    testData.drop(columns=CFG['catFeatures'], inplace=True)
    
    return testData


def trainModels(X_train, Y_train, makes, enc, CFG, hyperParams=None, toTrain=None, sampledNumFeatures=None, sampledCatFeatures=None):
    
    
    
    if toTrain is not None:
        makes = toTrain
        
    
    makeModels = {}
    for make, carModel in makes:
        if make not in makeModels:
            makeModels[make]={}
        
        
        if sampledCatFeatures is None:
            catFeatures = CFG['catFeatures']
        else:
            catFeatures = sampledCatFeatures
            
        if sampledNumFeatures is None:
            numFeatures = CFG['numFeatures']
        else:
            numFeatures = sampledNumFeatures
        
        
        make_X_train, make_Y_train = getMakeData(X=X_train, Y=Y_train, 
                                                make=make, carModel=carModel,
                                                numFeatures=numFeatures, catFeatures=catFeatures)
        
        
        
        encodedData = enc.transform(make_X_train[catFeatures])
        encodedFeatures = ['encoded_'+str(x) for x in range(len(encodedData[0]))]
        encodedData = pd.DataFrame(data=encodedData, 
                                   columns = encodedFeatures, 
                                  index = make_X_train.index)
        make_X_train = pd.concat([make_X_train, encodedData], axis=1)
        make_X_train.drop(columns=catFeatures, inplace=True)
        
        scalar = StandardScaler()
        selector = VarianceThreshold(threshold=CFG['variance_threshold'])
        pca = PCA(n_components=CFG['n_components'])
        
        if CFG['estimator'] == 'Ridge':
            model = Ridge(random_state=CFG['seed'])
        elif CFG['estimator'] == 'ElasticNet':
            model = ElasticNet(random_state=CFG['seed'])
        elif CFG['estimator'] == 'Lasso':
            model = Lasso(random_state=CFG['seed'])
        elif CFG['estimator'] == 'HuberRegressor':
            model = HuberRegressor()
        elif CFG['estimator'] == 'RANSACRegressor':
            model = RANSACRegressor()
        elif CFG['estimator'] == 'TheilSenRegressor':
            model = TheilSenRegressor(random_state=CFG['seed'])
        else:
            print('Chosen model not supported')
            quit()
        
        param_grid=CFG['param_grid']
        if hyperParams is not None:
            for param in hyperParams:
                if param in param_grid:
                    param_grid[param] = hyperParams[param][make][carModel]
        
        gridSearch = GridSearchCV(estimator=model, param_grid=param_grid,
                                  scoring=CFG['scoring'], refit=CFG['refit'],
                                  cv=CFG['cv'], verbose=CFG['verbose'],
                                  return_train_score=CFG['return_train_score'])
        
        
        pipe = Pipeline([('scaler', scalar), ('selector', selector), ('pca', pca), ('estimator', gridSearch)])
        pipe.fit(make_X_train, make_Y_train)
        makeModels[make][carModel]=pipe
    
    return makeModels
    


def testModels(X_test, Y_test, makeModels, enc, CFG, sampledNumFeatures=None, sampledCatFeatures=None):
    resultsDf = pd.DataFrame()
    for make in makeModels:
        for carModel in makeModels[make]:
            pipe = makeModels[make][carModel]  
            
            if sampledCatFeatures is None:
                catFeatures = CFG['catFeatures']
            else:
                catFeatures = sampledCatFeatures
                
            if sampledNumFeatures is None:
                numFeatures = CFG['numFeatures']
            else:
                numFeatures = sampledNumFeatures
            
            
            make_X_test, make_Y_test = getMakeData(X=X_test, Y=Y_test, 
                                                    make=make, carModel=carModel,
                                                    numFeatures=numFeatures, catFeatures=catFeatures)
                
            encodedData = enc.transform(make_X_test[catFeatures])
            encodedFeatures = ['encoded_'+str(x) for x in range(len(encodedData[0]))]
            encodedData = pd.DataFrame(data=encodedData, 
                                       columns = encodedFeatures, 
                                       index = make_X_test.index)
            make_X_test = pd.concat([make_X_test, encodedData], axis=1)
            make_X_test.drop(columns=catFeatures, inplace=True)
            
            
            Y_predict = pipe.predict(make_X_test)
            score = pipe.score(make_X_test, make_Y_test)
            
            outputDf = pd.DataFrame()
            outputDf['true'] = make_Y_test
            outputDf['predict'] = pd.Series(Y_predict, index=make_Y_test.index)
            outputDf['make'] = make
            outputDf['model'] = carModel
            outputDf['score'] = score
            
            bestParams = pipe.named_steps['estimator'].best_params_
            for param in bestParams:
                outputDf[param] = bestParams[param]
                
            if CFG['output_coef']:
                estimator = pipe.named_steps['estimator'].best_estimator_
                if 'coef_' in dir(estimator):
                    for idx, coef in enumerate(estimator.coef_):
                        outputDf['coef'+str(idx)] = coef
                if 'intercept_' in dir(estimator):
                    outputDf['intercept_'] = estimator.intercept_
                
            
            
            if len(resultsDf)==0:
                resultsDf = outputDf.copy()
            else:
                resultsDf = pd.concat([resultsDf, outputDf])
    
    return resultsDf
            

        
def writeToDisk(trainedModels, testResults, CFG):
    
    pathToWrite = Path.cwd()
    pathToWrite = pathToWrite.joinpath('Models')
    now = datetime.now()
    now = now.strftime('%Y%m%d%H%M')
    dirName = CFG['estimator']+'_'+now
    pathToWrite = pathToWrite.joinpath(dirName)
    
    if not(pathToWrite.exists()):
        pathToWrite.mkdir()
    
    with pathToWrite.joinpath('CFG.txt').open(mode='w') as f:
        f.write(json.dumps(CFG))
    
    cvResults = pd.DataFrame()
    for make in trainedModels:
        for carModel in trainedModels[make]:
            pipe = trainedModels[make][carModel]
            gridSearch = pipe.named_steps['estimator']
            pathToDump = pathToWrite.joinpath(make+'_'+carModel+'.joblib')
            dump(gridSearch, str(pathToDump))
            
            gridSearchCv = pd.DataFrame(gridSearch.cv_results_)
            gridSearchCv['make'] = make
            gridSearchCv['model'] = carModel
            if len(cvResults)==0:
                cvResults = gridSearchCv.copy()
            else:
                cvResults = pd.concat([cvResults, gridSearchCv])
                
    cvResults.to_csv(str(pathToWrite.joinpath('CV.csv')))
    testResults.to_csv(str(pathToWrite.joinpath('testResults.csv')))

def writeSamplesToDisk(samplesResults, CFG):
    
    pathToWrite = Path.cwd()
    pathToWrite = pathToWrite.joinpath('Models')
    now = datetime.now()
    now = now.strftime('%Y%m%d%H%M')
    dirName = CFG['estimator']+'features_sampled'+'_'+now
    pathToWrite = pathToWrite.joinpath(dirName)
    
    if not(pathToWrite.exists()):
        pathToWrite.mkdir()
    
    with pathToWrite.joinpath('CFG.txt').open(mode='w') as f:
        f.write(json.dumps(CFG))
    
    for idx, modelOutput  in enumerate(samplesResults):
        testResults, trainedModels, sample = modelOutput
        testResults['features'] = str(sample)
        
        cvResults = pd.DataFrame()
        for make in trainedModels:
            for carModel in trainedModels[make]:
                pipe = trainedModels[make][carModel]
                gridSearch = pipe.named_steps['estimator']
                pathToDump = pathToWrite.joinpath(make+'_'+carModel+str(idx)+'.joblib')
                dump(gridSearch, str(pathToDump))
                
                gridSearchCv = pd.DataFrame(gridSearch.cv_results_)
                gridSearchCv['make'] = make
                gridSearchCv['model'] = carModel
                if len(cvResults)==0:
                    cvResults = gridSearchCv.copy()
                else:
                    cvResults = pd.concat([cvResults, gridSearchCv])
                    
        cvResults.to_csv(str(pathToWrite.joinpath(str(idx)+'CV.csv')))
        testResults.to_csv(str(pathToWrite.joinpath(str(idx)+'testResults.csv')))


if CFG['csv'] == True:
    carData = pd.read_csv(CFG['inputCsv'])

valuationDate = CFG['valuationDate']
valuationDate = date(int(valuationDate[:4]), int(valuationDate[5:7]), int(valuationDate[-2:]))

monthMapping = {'January':	1,
                'February':	2,
                'March':	3,
                'April':	4,
                'May':	5,
                'June':	6,
                'July':	7,
                'August':	8,
                'September':	9,
                'October':	10,
                'November':	11,
                'December':	12,
                'None': None}

hyperParams = None
if CFG['readHyperParams'] is not None:
    with open(CFG['readHyperParams']) as f:
        hyperParams = json.loads(f.read())
        
toTrain = None
if CFG['toTrain'] is not None:
    with open(CFG['toTrain']) as f:
        toTrain = json.loads(f.read())

carData = carData[~carData[CFG['target']].isna()]
carData['price'] = carData['price'].apply(lambda price:log(price))

carData['age'] = (valuationDate.year - carData['vehicle_year'])  

carData['nct_month'] = carData['nct_month'].apply(lambda month: monthMapping[month])
carData['nct_remaining'] =  (carData['nct_year'] - valuationDate.year)
carData['nct_remaining'] = carData['nct_remaining'] + ((carData['nct_month'] - valuationDate.month)/12)
carData['nct_remaining'] = carData['nct_remaining'].where(~(carData['nct_year'].isna() | carData['nct_month'].isna()), -1)
  
carData['make_and_model'] = carData['make']+' '+carData['model']
makes = getDistinctMakes(carData)

carData = fillna(carData=carData,CFG=CFG)



featuresToSample = CFG['numFeatures'] + CFG['catFeatures']
featuresToSample.remove('age')
featuresToSample.remove('mileage')
featuresToSample.remove('transmission')
featuresToSample.remove('engine')
samples = []
for k in range(len(featuresToSample)):
    samples.extend(combinations(featuresToSample, k))
samples.remove(())

samplesResults = []
for sample in tqdm(samples):

    sampledNumFeatures = ['age',  'mileage', 'engine']
    sampledCatFeatures = ['transmission']
    for feature in sample:
        if feature in CFG['numFeatures']:
            sampledNumFeatures.append(feature)
        elif feature in CFG['catFeatures']:
            sampledCatFeatures.append(feature)
    
    
    databaseConfig = None
    categories = []
    with open('dataSourcing/configs/databaseConfig.yaml', 'r') as f:
        databaseConfig = yaml.full_load(f)
    for column in sampledCatFeatures:
        categories.append(irishCarPriceDatabase.selectDistinctValues(databaseConfig, column))
    
    
        
    
    enc = OneHotEncoder(drop=CFG['enc_drop'], 
                        sparse=CFG['enc_sparse'], 
                        handle_unknown=CFG['enc_handle_unknown'],
                        categories=categories)
    enc.fit(carData[sampledCatFeatures])

    X_train, X_test, Y_train, Y_test = train_test_split(carData,
                                                        carData[CFG['target']],
                                                        test_size=CFG['test_size'],
                                                        stratify=carData['make_and_model'],
                                                        random_state=CFG['seed'])
    
    rng = random.default_rng()
    mileageStd = X_train[X_train['mileage']>0]['mileage'].std()
    
    noise = rng.normal(loc=CFG['annual_mileage'],scale=mileageStd, size=len(X_train))
    X_train['mileage_fill_values'] =  (X_train['age']-1)*CFG['annual_mileage'] + noise
    X_train['replace_mileage'] = X_train['mileage']/X_train['age']
    X_train['replace_mileage'] = ((X_train['replace_mileage'] < CFG['annual_mileage_cutoff']) & (X_train['age']>=1))
    
    X_train['mileage'] = X_train['mileage'].where(~X_train['replace_mileage'], X_train['mileage_fill_values'])
    
    noise = rng.normal(loc=CFG['annual_mileage'], scale=mileageStd, size=len(X_test))
    X_test['mileage_fill_values'] =  (X_test['age']-1)*CFG['annual_mileage'] + noise
    X_test['replace_mileage'] = X_test['mileage']/X_test['age']
    X_test['replace_mileage'] = ((X_test['replace_mileage'] < CFG['annual_mileage_cutoff']) & (X_test['age']>=1))
    X_test['mileage'] = X_test['mileage'].where(~X_test['replace_mileage'], X_test['mileage_fill_values'])
    
    
  
    trainedModels = trainModels(X_train=X_train, Y_train=Y_train, makes=makes, enc=enc, CFG=CFG, 
                                hyperParams=hyperParams, toTrain=toTrain, sampledNumFeatures=sampledNumFeatures, 
                                sampledCatFeatures=sampledCatFeatures)
    
    testResults = testModels(X_test=X_test, Y_test=Y_test, makeModels=trainedModels,  
                             enc=enc, CFG=CFG, sampledNumFeatures=sampledNumFeatures, 
                             sampledCatFeatures=sampledCatFeatures)
    
    samplesResults.append((testResults, trainedModels, sample)) 
    

writeSamplesToDisk(samplesResults=samplesResults, CFG=CFG)
