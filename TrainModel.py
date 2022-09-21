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
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error


from sklearn.linear_model import Ridge, Lasso, ElasticNet, HuberRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from numpy import random




CFG = {
       #IO Parameters
       'inputCsv': 'cleandata.csv',
       'outputPath': 'Models',
       'notes': 'Change PCA regression to 1 component',
       
       #Numerical, Categorical and Target variables
       'numFeatures': ['age', 'mileage', 'engine', 'mpg', 'owners', 'nct_remaining', 'tax'],
       'catFeatures': ['transmission', 'body', 'color', 'fuel'],
       'target': 'price',
       
       #Date to value to cars, should be close to the date the data was sourced
       'valuationDate': '2022-09-07',
       
       #The average annual km a car drives in Ireland. This will be used to fill missing values
       'annual_mileage': 17000,
       
       #If a car older than 1 year has <1000 km driven then mileage will be treated as NULL.
       'annual_mileage_cutoff': 1000,
       
       #Method used to fill missing values for variables other than mileage.
       'fillna_method': 'median',
       
       
       #Size of the test set
       'test_size': 0.3,
       
       #OneHot Encoder Parameters
       'enc_drop': 'first',
       'enc_sparse': False,
       'enc_handle_unknown': 'ignore',
       
       #PCA Parameters for the linear regression
       'n_components_LinearRegression': 1,
       'n_components_Model': 0.99,
       
       #GridSearchCV Parameters
       'linearRegressor': 'Lasso',
       'linearRegressorFeatures': ['age', 'mileage'],
       'linearRegressorFitIntercept': True,
       'alpha': [58],
       'outputRegressorCoeff': True,
       'estimator': 'GradientBoostingRegressor',
       'param_grid': {'n_estimators': [600],
                      'max_depth': [6],
                      'min_samples_split': [6],
                      'min_samples_leaf': [9],
                      'min_impurity_decrease': [0],
                      'min_weight_fraction_leaf': [0],
                      'max_features': [0.7],
                      'max_leaf_nodes': [12],
                      'ccp_alpha': [0]
                      },
       'scoring': 'neg_mean_squared_error',
       'refit': True,
       'cv': 5,
       'verbose':1,
       'return_train_score': True,
       'output_coef': False,
       'seed': 1023,
       'readHyperParams': None,
       
       'toTrain': None
       }


def fillna(carData, CFG):
    """
    Fill missing values using the median of each column.
    The median over the car model is used as the fill value. If this median doesn't exist
    then the median over the car make is used. If this also doesn't exist then the median of the overall
    column is used
    
    Parameters
    ----------
    carData : DataFrame
        The dataset containg the car data. The DataFrame is assumed to contain 'model'
        and 'make' as columns
    CFG : dict
        dictionary containing the configuration for this fit.

    Returns
    -------
    carData : DataFrame
        Returns the dataset passed but with missing values filled
        
    fillValues : DataFrame
        The values used to fill the missing data

    """

    medianModelData = carData.groupby(by=['make','model'])[CFG['numFeatures']].median()
    medianMakeData = carData.groupby(by='make')[CFG['numFeatures']].median()
    medianData = carData[CFG['numFeatures']].median()
    
    carData=carData.merge(medianModelData[CFG['numFeatures']], how='left', 
                          left_on=['make', 'model'], right_index=True)
    
    for feature in CFG['numFeatures']:
        carData[feature] =  carData[feature+'_x'].fillna(carData[feature+'_y'])
        carData.drop(labels = [feature+'_x', feature+'_y'], axis=1, inplace=True)

    carData=carData.merge(medianMakeData[CFG['numFeatures']], how='left', 
                          left_on='make', right_index=True)
    for feature in CFG['numFeatures']:
        carData[feature] =  carData[feature+'_x'].fillna(carData[feature+'_y'])
        carData.drop(labels = [feature+'_x', feature+'_y'], axis=1, inplace=True)    

    carData.fillna(medianData, inplace=True)

    fillValues = medianModelData.reset_index().copy()
    fillValues = pd.concat([fillValues, medianMakeData.reset_index()])
    
    medianData = pd.DataFrame(data = [medianData.values], columns=medianData.index)
    fillValues = pd.concat([fillValues, medianData])
        
    return carData, fillValues


def removeMileageOutlier(X, std, CFG):
    """
    Remove outliers in the mileage column. An outlier is identified as any datapoint with
    X['mileage']/X['age ']< CFG['annual_mileage_cutoff'] & X['age']>=1
    
    The fill value is calulated as 
    (X['age']-1)*CFG['annual_mileage'] + noise
    where noise is sampled from a Gaussian distribution with 
    mean = CFG['annual_mileage_cutoff']
    std = std
    
    Parameters
    ----------
    X : DataFrame
        Feature dataset. Assumed to contain 'age' and 'mileage' as columns
    std : float
        The standaed deviation of the mileage column
    CFG : dict
        dictionary containing the configuration for this fit.

    Returns
    -------
    X : DataFrame
        Returns the DataFrame passed with mileage outliers removed

    """
    rng = random.default_rng()
    noise = rng.normal(loc=CFG['annual_mileage'],scale=std, size=len(X))
    
    X['mileage_fill_values'] =  (X['age']-1)*CFG['annual_mileage'] + noise
    X['replace_mileage'] = X['mileage']/X['age']
    X['replace_mileage'] = ((X['replace_mileage'] < CFG['annual_mileage_cutoff']) & (X['age']>=1))
    
    X['mileage'] = X['mileage'].where(~X['replace_mileage'], X['mileage_fill_values'])
    
    return X
    

def getMakeData(X, Y, make, carModel, features):
    """
    Get the sample from X and Y for a specifc make and carModel. The samples from X will
    only have the columns specified by features.

    Parameters
    ----------
    X : DataFrame
        Feature dataset. Assumded to contain 'make' and 'model' as columns
    Y : Series
        Target variable
    make : string
        The make of car to retrieve data for
    carModel : string
        The car model to retrieve data for
    features : list
        The list of features in X to return

    Returns
    -------
    make_X : DataFrame
        Returns the a DataFrame whose columns are features for the make and carModel passed
    make_Y : TYPE
        Returns target variable for the make and carModel passed.

    """
    make_X= X[(X['model']==carModel)  & (X['make']==make)]
    make_X = make_X[features]
    make_Y = Y[make_X.index]
    return make_X, make_Y


def getDistinctMakes(carData):
    """
    Get the distinct make and models found in the carData DataFrame

    Parameters
    ----------
    carData : DataFrame
        The dataset containg the car data. The DataFrame is assumed to contain 'model'
        and 'make' as columns

    Returns
    -------
    makes : list of tuples
        Returns a list where each entry is a tuple like (make, model) e.g.
        [('Ford', 'Focus'), ('Nissan', 'Micra'),...]

    """
    makes=[]
    for make in carData[['make', 'model']].values:
        if tuple(make) not in makes:
            makes.append(tuple(make))
    return makes
        

def trainModel(X_train, Y_train, makes, enc, CFG, alpha=1):
    """
    

    Parameters
    ----------
    X_train : DataFrame
        DataFrame containing the feature training data. The DataFrame is required to have 'make' and 'model' as
        columns as well as all the columns seen in CFG['numFeatures'] and CFG['catFeatures']. Additional columns
        can be passed during training but they will be ignored.
    Y_train : Series
        Target variable of the training set
    makes : list of tuples
        list containing all unique pairs of make and model seen during training, 
        e.g. [('Ford', 'Focus'), ('Nissan', 'Micra'), ..]
    enc : OneHotEncoder
        An instance of the OnHotEncoder class initalized befroe training.
    CFG : dict
        dictionary containing the configuration for this fit.
    alpha : float, optional
        hyperparameter for the liner regression used for feature engineering. The default is 1.

    Returns
    -------
    pipe : Pipeline()
        The trained model
    regressorCoeff : DataFrame
        A DataFrame containg the coefficients of the linear regressor calculated during training
    enc: OneHotEncoder
        The encoder fitted on the training data
    Y_predict : Series
        The predicted values of the train dataset after fitting

    """
    
    catFeatures = CFG['catFeatures'].copy()
    numFeatures = CFG['numFeatures'].copy()
    
    regressorCoeff = []
    makeEntries = []
    carModelEntries = []
    for make, carModel in makes:
        
        make_X_train, make_Y_train = getMakeData(X=X_train, Y=Y_train, 
                                                make=make, carModel=carModel,
                                                features=CFG['linearRegressorFeatures'])
        
        if CFG['linearRegressor'] == 'Ridge':
            linearRegressor = Ridge(random_state=CFG['seed'], 
                                    fit_intercept=CFG['linearRegressorFitIntercept'])
        elif CFG['linearRegressor'] == 'ElasticNet':
            linearRegressor = ElasticNet(random_state=CFG['seed'],
                                         fit_intercept=CFG['linearRegressorFitIntercept'])
        elif CFG['linearRegressor'] == 'Lasso':
            linearRegressor = Lasso(random_state=CFG['seed'],
                                    fit_intercept=CFG['linearRegressorFitIntercept'])
        elif CFG['linearRegressor'] == 'HuberRegressor':
            linearRegressor = HuberRegressor(alpha=alpha,
                                             fit_intercept=CFG['linearRegressorFitIntercept'])
        else:
            print('Chosen linear regressor not supported')
            quit()
        
        linearRegressorPipe = Pipeline([('regressorScaler', StandardScaler()), 
                                       ('regressorPCA', PCA(n_components=CFG['n_components_LinearRegression'])),
                                       ('regressor', linearRegressor)])
        
        
        linearRegressorPipe.fit(make_X_train, make_Y_train)
        
        regressorFeatures = linearRegressorPipe.named_steps['regressor'].coef_
        regressorFeatures = list(regressorFeatures)
        regressorFeatures.append(linearRegressorPipe.named_steps['regressor'].intercept_)
        
        makeEntries.append(make)
        carModelEntries.append(carModel)
        regressorCoeff.append(regressorFeatures)
        

    regressorColumns = []
    for idx in range(len(regressorCoeff[0])):
        regressorColumns .append('regessorFeature'+str(idx))
    numFeatures.extend(regressorColumns)
    
    regressorCoeff = pd.DataFrame(data=regressorCoeff, columns=regressorColumns)
    regressorCoeff['make'] = pd.Series(makeEntries)
    regressorCoeff['model'] = pd.Series(carModelEntries)

    X_train = X_train.merge(right=regressorCoeff, how='left', 
                            left_on=['make', 'model'], 
                            right_on=['make', 'model'])
    X_train.index = Y_train.index
    
    
    encodedData = enc.fit_transform(X_train[catFeatures])
    encodedFeatures = ['encoded_'+str(x) for x in range(len(encodedData[0]))]
    encodedData = pd.DataFrame(data=encodedData, 
                               columns = encodedFeatures, 
                               index = X_train.index)
    X_train = pd.concat([X_train, encodedData], axis=1)
    X_train.drop(columns=catFeatures, inplace=True)
    
    scalar = StandardScaler()
    pca = PCA(n_components=CFG['n_components_Model'])
    
    if CFG['estimator'] == 'RandomForestRegressor':
        estimator = RandomForestRegressor()
    elif CFG['estimator'] == 'ExtraTreesRegressor':
        estimator = ExtraTreesRegressor()
    elif CFG['estimator'] == 'GradientBoostingRegressor':
        estimator = GradientBoostingRegressor()
    elif CFG['estimator'] == 'HistGradientBoostingRegressor':
        estimator = HistGradientBoostingRegressor()
    
    else:
        print('Chosen linear regressor not supported')
        quit()
    
    
    gridSearch = GridSearchCV(estimator=estimator, param_grid=CFG['param_grid'],
                              scoring=CFG['scoring'], refit=CFG['refit'],
                              cv=CFG['cv'], verbose=CFG['verbose'],
                              return_train_score=CFG['return_train_score'])
    
    pipe = Pipeline([('scaler', scalar), ('pca', pca), ('estimator', gridSearch)])
    X_train = X_train[numFeatures+encodedFeatures]
    pipe.fit(X_train, Y_train)
    Y_predict = pipe.predict(X_train)
    
    return (pipe, regressorCoeff, enc), Y_predict
    


def testModel(X_test, Y_test, trainResults, CFG):
    """
    

    Parameters
    ----------
    X_test : DataFrame
        DataFrame containing the feature test data. The DataFrame is required to have 'make' and 'model' as
        columns as well as all the columns seen in CFG['numFeatures'] and CFG['catFeatures']. Additional columns
        can be passed during testing but they will be ignored.
    Y_test : Series
        Target variable of the training set.
    trainResults : tuple
        Output from trainModel of the form (Pipeline(), DataFrame(), OneHotEncoder()) where
        Pipeline() is the trained model, DataFrame contains the coefficients of the linear regressor calculated
        during training and OneHotEncoder() is the encoder fitted during training
    CFG : dict
        dictionary containing the configuration for this fit.

    Returns
    -------
    outputDf : DataFrame
        Contains the true and predicted value of each sample along with the score of the dataset.

    """
    pipe, regressorCoeff, enc = trainResults
    
    catFeatures = CFG['catFeatures'].copy()
    numFeatures = CFG['numFeatures'].copy()
        
    
    X_test = X_test.merge(regressorCoeff, left_on=['make', 'model'], right_on=['make', 'model'], how='left')
    X_test.index=Y_test.index
    
    regressorFeatures = list(regressorCoeff.columns)
    regressorFeatures.remove('make')
    regressorFeatures.remove('model')
    numFeatures.extend(regressorFeatures)
    
    
    encodedData = enc.transform(X_test[catFeatures])
    encodedFeatures = ['encoded_'+str(x) for x in range(len(encodedData[0]))]
    encodedData = pd.DataFrame(data=encodedData, 
                               columns = encodedFeatures, 
                               index = X_test.index)
    X_test = pd.concat([X_test, encodedData], axis=1)
    X_test.drop(columns=catFeatures, inplace=True)
    
    
    outputDf = pd.DataFrame(index=Y_test.index)
    outputDf['id'] = X_test['page_id']
    outputDf['make'] = X_test['make']
    outputDf['model'] = X_test['model']
    

    X_test = X_test[numFeatures+encodedFeatures]
    
    Y_predict = pipe.predict(X_test)
    score = pipe.score(X_test, Y_test)
    
    outputDf['true'] = Y_test
    outputDf['predict'] = pd.Series(Y_predict, index=Y_test.index)
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

    return outputDf
            

        
def writeToDisk(trainResults, testResults, CFG, fillValues = None):
    """
    

    Parameters
    ----------
    trainResults : tuple
        Output from trainModel of the form (Pipeline(), DataFrame(), OneHotEncoder()) where
        Pipeline() is the trained model, DataFrame contains the coefficients of the linear regressor calculated
        during training and OneHotEncoder() is the encoder fitted during training
    testResults : DataFrame
        Contains the true and predicted value of each sample along with the score of the dataset.
    fillValues : DataFrame(optional)
        The values used to fill missing values in the data set
    CFG : dict
        dictionary containing the configuration for this fit.

    Returns
    -------
    None.

    """
    pathToWrite = Path.cwd()
    pathToWrite = pathToWrite.joinpath('Models')
    now = datetime.now()
    now = now.strftime('%Y%m%d%H%M')
    dirName = CFG['linearRegressor']+'_'+CFG['estimator']+'_'+now
    pathToWrite = pathToWrite.joinpath(dirName)
    
    if not(pathToWrite.exists()):
        pathToWrite.mkdir()
    
    with pathToWrite.joinpath('CFG.txt').open(mode='w') as f:
        f.write(json.dumps(CFG))
    
    
    cvResults = pd.DataFrame()
    pipe, coeff, enc = trainResults
    pathToDump = pathToWrite.joinpath('estimator.joblib')
    dump(pipe, str(pathToDump))
    
    pathToDump = pathToWrite.joinpath('encoder.joblib')
    dump(enc, str(pathToDump))
    
    gridSearch = pipe.named_steps['estimator']
    gridSearchCv = pd.DataFrame(gridSearch.cv_results_)
    if len(cvResults)==0:
        cvResults = gridSearchCv.copy()
    else:
        cvResults = pd.concat([cvResults, gridSearchCv])
                
    cvResults.to_csv(str(pathToWrite.joinpath('CV.csv')), index=False)
    testResults.to_csv(str(pathToWrite.joinpath('testResults.csv')))
    if CFG['outputRegressorCoeff']:
        coeff.to_csv(str(pathToWrite.joinpath('regressorCoeff.csv')), index=False)
    if fillValues is not None:
        fillValues.to_csv(str(pathToWrite.joinpath('fillValues.csv')), index=False)


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


carData = carData[~carData[CFG['target']].isna()]
carData['price'] = carData['price'].apply(lambda price:log(price))

carData['age'] = (valuationDate.year - carData['vehicle_year'])  

carData['nct_month'] = carData['nct_month'].apply(lambda month: monthMapping[month])
carData['nct_remaining'] =  (carData['nct_year'] - valuationDate.year)
carData['nct_remaining'] = carData['nct_remaining'] + ((carData['nct_month'] - valuationDate.month)/12)
carData['nct_remaining'] = carData['nct_remaining'].where(~(carData['nct_year'].isna() | carData['nct_month'].isna()), -1)
  
carData['make_and_model'] = carData['make']+' '+carData['model']
makes = getDistinctMakes(carData)

carData, fillValues = fillna(carData=carData,CFG=CFG)


databaseConfig = None
categories = []
with open('dataSourcing/configs/databaseConfig.yaml', 'r') as f:
    databaseConfig = yaml.full_load(f)
for column in CFG['catFeatures']:
    categories.append(irishCarPriceDatabase.selectDistinctValues(databaseConfig, column))

enc = OneHotEncoder(drop=CFG['enc_drop'], 
                    sparse=CFG['enc_sparse'], 
                    handle_unknown=CFG['enc_handle_unknown'],
                    categories=categories)

X_train, X_test, Y_train, Y_test = train_test_split(carData,
                                                    carData[CFG['target']],
                                                    test_size=CFG['test_size'],
                                                    stratify=carData['make_and_model'],
                                                    random_state=CFG['seed'])


mileageStd = X_train[X_train['mileage']>0]['mileage'].std()
X_train = removeMileageOutlier(X=X_train, std=mileageStd, CFG=CFG)
X_test = removeMileageOutlier(X=X_test, std=mileageStd, CFG=CFG)

alltestResults = pd.DataFrame()
for alpha in tqdm(CFG['alpha']):
    trainResults, Y_predict = trainModel(X_train=X_train, Y_train=Y_train, makes=makes, 
                                         enc=enc, CFG=CFG, alpha=alpha)
    
    trainScore = mean_squared_error(Y_predict, Y_train)
    
    testResults = testModel(X_test=X_test, Y_test=Y_test, trainResults=trainResults, CFG=CFG)
    
    testResults['alpha_value'] = alpha
    testResults['trainScore'] = trainScore
    if len(alltestResults)==0:
        alltestResults = testResults.copy()
    else:
        alltestResults = pd.concat([alltestResults, testResults])
writeToDisk(trainResults=trainResults, testResults=alltestResults, fillValues=fillValues, CFG=CFG)
