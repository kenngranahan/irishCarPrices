# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 10:41:34 2022

@author: User
"""



import warnings
from pathlib import Path
from joblib import load
import json


import pandas as pd
from datetime import date
from math import exp


class carPricePredictor:
    """
    A class used to price cars based on the car data provied by the user
    
    """
    def __init__(self):
        self._modelName = 'Lasso_GradientBoostingRegressor_202209211352'
            
        self._modelPath = Path.cwd().joinpath('Models\\'+self._modelName)
        
        
        with self._modelPath.joinpath('CFG.txt').open() as f:
            self._CFG = json.loads(f.read())
        
        self._fillValues = pd.read_csv(str(self._modelPath.joinpath('fillValues.csv')))
        self._regressorCoeff=pd.read_csv(str(self._modelPath.joinpath('regressorCoeff.csv')))
        self._enc=load(str(self._modelPath.joinpath('encoder.joblib')))
        self._model=load(str(self._modelPath.joinpath('estimator.joblib')))
        
        
        self._expectedColumns = {'make': 'object',
                                    'model': 'object',
                                    'year': 'float64',
                                    'mileage': 'float64',
                                    'transmission': 'object',
                                    'engine': 'float64',
                                    'body': 'object',
                                    'fuel': 'object',
                                    'mpg': 'float64',
                                    'owners': 'float64',
                                    'color': 'object',
                                    'tax': 'float64',
                                    'nct_month': 'object',
                                    'nct_year': 'float64'}

        self._todaysDate = date.today()
        self._monthMapping = {'January':	1,
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
        
        
    def getModel(self):
        return self._modelName

    def getMonths(self):
        return self._monthMapping.keys()
    
    def getExpectedColumns(self):
        return self._expectedColumns


    def _engineerFeatures(self, X):

        X['age'] = (self._todaysDate.year - X['year'])  
        
        X['nct_month'] = X['nct_month'].apply(lambda month: self._monthMapping[month] if month is not None else None)
        X['nct_remaining'] =  (X['nct_year'] - self._todaysDate.year)
        X['nct_remaining'] = X['nct_remaining'] + ((X['nct_month'] - self._todaysDate.month)/12)
        X['nct_remaining'] = X['nct_remaining'].where(~(X['nct_year'].isna() | X['nct_month'].isna()), -1)

        X['mileage_fill_values'] =  (X['age']-1)*self._CFG['annual_mileage']
        X['replace_mileage'] = X['mileage']/X['age']
        X['replace_mileage'] = ((X['replace_mileage'] < self._CFG['annual_mileage_cutoff']) & (X['age']>=1))
        X['mileage'] = X['mileage'].where(~X['replace_mileage'], X['mileage_fill_values'])

        return X        
    
    def _getRegressorCoeff(self, X):
        
        X = X.merge(right=self._regressorCoeff, how='left', 
                    left_on=['make', 'model'], 
                    right_on=['make', 'model'])
        makeMeanCoeff = self._regressorCoeff.drop('model', axis=1)
        makeMeanCoeff = makeMeanCoeff.groupby('make').mean()
        X=X.merge(makeMeanCoeff, how='left', left_on='make', right_index=True)
        
        for column in self._regressorCoeff:
            if column != 'make' and column != 'model':
                X[column] = X[column+'_x'].fillna(X[column+'_y'])
                X.drop(labels = [column+'_x', column+'_y'], axis=1, inplace=True)    
                
        meanCoeff = self._regressorCoeff.drop(['make', 'model'], axis=1)
        meanCoeff = meanCoeff.mean()
        X.fillna(meanCoeff, inplace=True)
        
        return X
    
 
    def _fillMissingValue(self, X):
        medianModelData = self._fillValues[~self._fillValues['model'].isna()]
        medianMakeData = self._fillValues[(~self._fillValues['make'].isna()) & self._fillValues['model'].isna()]
        medianData = self._fillValues[self._fillValues['make'].isna()]
        
        X=X.merge(medianModelData, how='left', 
                              left_on=['make', 'model'],
                              right_on=['make', 'model'])        
        for feature in medianModelData.columns:
            if feature!='make' and feature!='model':
                X[feature] =  X[feature+'_x'].fillna(X[feature+'_y'])
                X.drop(labels = [feature+'_x', feature+'_y'], axis=1, inplace=True)

        X=X.merge(medianMakeData, how='left', 
                              left_on=['make', 'model'],
                              right_on=['make', 'model'])
        for feature in medianMakeData.columns:
            if feature!='make' and feature!='model':
                X[feature] =  X[feature+'_x'].fillna(X[feature+'_y'])
                X.drop(labels = [feature+'_x', feature+'_y'], axis=1, inplace=True)

        X.fillna(medianData, inplace=True)

        return X
    
    
    def _checkXInput(self, X):
        
        if not(isinstance(X, pd.DataFrame)):
            raise TypeError('Expected X to be an instance of pd.DataFrame()')
        
        missingColumns = []
        for column in self._expectedColumns:
            if column not in X.columns:
                missingColumns.append(column)
        if missingColumns!=[]:
            raise ValueError('X is missing columns {}'.format(missingColumns))
        
        
        for column in self._expectedColumns:
            try:
                X = X.astype({column: self._expectedColumns[column]})
            except:
                raise ValueError('Failed to cast {} as type {}'.format(column, self._expectedColumns[column]))
                
        for month in X['nct_month'].unique():
            if month not in self._monthMapping:
                raise ValueError('Unrecognized month {} found in nct_month'.format(month))
        
        if X[['make', 'model']].isna().sum().sum()!=0:
            X = X[(~X['make'].isna()) & (~X['model'].isna())]
            warnings.warn('NA values were found in make or model columns. These rows will be dropped')
    
        return X
    
    def predictPrice(self, X):
        """
        

        Parameters
        ----------
        X : Pandas DataFrame
            A DataFrame containing the car data. The DataFrame is expected to contain
            the columns returned by the getExpectedColumns() method. The columns
            'make' and 'model' are both expected to be none empty, any rows with missing values
            in either column will be dropped.
            The column 'nct_month' is expected to contain months formated as the ones returned
            by the getMonths() method

        Returns
        -------
        Y : Pandas Series
            Returns a Series containing the predicted prices with the same index as X.

        """
        
        X = self._checkXInput(X)

        X = self._engineerFeatures(X)
        X = self._fillMissingValue(X)
        X = self._getRegressorCoeff(X)
        
        encodedData = self._enc.transform(X[self._CFG['catFeatures']])
        encodedFeatures = ['encoded_'+str(x) for x in range(len(encodedData[0]))]
        encodedData = pd.DataFrame(data=encodedData, 
                                   columns = encodedFeatures, 
                                   index = X.index)
        X = pd.concat([X, encodedData], axis=1)
        X.drop(columns=self._CFG['catFeatures'], inplace=True)
        
        regressorColumns = list(self._regressorCoeff.drop(['make', 'model'], axis=1).columns.values)
        X = X[self._CFG['numFeatures']+regressorColumns+encodedFeatures]
        Y = self._model.predict(X)
        Y = pd.Series(data=Y, index=X.index)
        
        # The model was trained on log(price) so we need to apply exp() 
        # to the model output to determine the price 
        Y = Y.apply(exp)
        
        return Y

 