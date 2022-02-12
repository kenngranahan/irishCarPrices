
import datetime
import unittest
import json
import numpy as np
import yaml
import irishCarPriceDatabase
from pathlib import Path


class CarsIrelandScrappedData(unittest.TestCase):
    def test_deleteInsertSelect(self):
        with open('dataSourcing/databaseConfig.yaml', 'r') as f:
            databaseConfig = yaml.full_load(f)
        testConfig = databaseConfig['testConfig']

        irishCarPriceDatabase.deleteCarsIrelandScrappedData(
            downloadDate=testConfig['downloadDate'], databaseConfig=databaseConfig)

        with Path(testConfig['carsIrelandScrappedData'] + 'input.txt').open('r') as f:
            testInputData = json.loads(f.read())
        with Path(testConfig['carsIrelandScrappedData'] + 'output.txt').open('r') as f:
            testOutputData = json.loads(f.read())

        for id in testInputData:
            irishCarPriceDatabase.insertCarsIrelandScrappedData(
                testInputData[id], databaseConfig)
        queriedData = irishCarPriceDatabase.selectCarsIrelandScrappedData(
            testConfig['downloadDate'], databaseConfig)

        for id in queriedData:
            for column in queriedData[id]:
                queriedValue = queriedData[id][column]

                if queriedValue is None:
                    self.assertEqual(
                        "", testOutputData[id][column])
                else:
                    self.assertEqual(
                        queriedData[id][column], testOutputData[id][column])


class CarsIrelandCleanData(unittest.TestCase):
    def test_deleteInsertSelect(self):
        with open('dataSourcing/databaseConfig.yaml', 'r') as f:
            databaseConfig = yaml.full_load(f)
        testConfig = databaseConfig['testConfig']

        irishCarPriceDatabase.deleteCarsIrelandCleanData(
            downloadDate=testConfig['downloadDate'], databaseConfig=databaseConfig)

        with Path(testConfig['carsIrelandCleanData'] + 'input.txt').open('r') as f:
            testInputData = json.loads(f.read())
        with Path(testConfig['carsIrelandCleanData'] + 'output.txt').open('r') as f:
            testOutputData = json.loads(f.read())

        for id in testInputData:
            irishCarPriceDatabase.insertCarsIrelandCleanData(
                testInputData[id], databaseConfig)
        queriedData = irishCarPriceDatabase.selectCarsIrelandCleanData(
            testConfig['downloadDate'], databaseConfig)

        for id in testOutputData:
            for column in testOutputData[id]:
                queriedValue = queriedData[id][column]
                if (isinstance(queriedValue, float) and np.isnan(queriedValue)):
                    self.assertTrue(np.isnan(testOutputData[id][column]))
                elif queriedValue == '':
                    self.assertEqual(testOutputData[id][column], "")
                elif isinstance(queriedValue, datetime.date):
                    self.assertEqual(
                        queriedValue, datetime.date.fromisoformat(testOutputData[id][column]))
                else:
                    self.assertEqual(
                        queriedValue, testOutputData[id][column])
