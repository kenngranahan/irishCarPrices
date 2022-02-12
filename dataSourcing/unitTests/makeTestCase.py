from pathlib import Path
from numpy import isnan
import yaml
import json
import dataProcesser
import irishCarPriceDatabase


def makeCarsIrelandCleanData():
    with Path('dataSourcing/databaseConfig.yaml').open('r') as f:
        databaseConfig = yaml.full_load(f)
    testConfig = databaseConfig['testConfig']

    with Path(testConfig['carsIrelandScrappedData']+'input.txt').open('r') as f:
        scrappedData = json.loads(f.read())

    cleanData = dataProcesser.cleanRawData(scrappedData)
    for id in cleanData:
        cleanData[id]['download_date'] = testConfig['downloadDate']
        cleanData[id]['id'] = id
    with Path(testConfig['carsIrelandCleanData']+'input.txt').open('w') as f:
        f.write(json.dumps(cleanData))

    with Path(testConfig['carsIrelandCleanData']+'output.txt').open('w') as f:
        f.write(json.dumps(cleanData))

    return None


def deleteData():
    with Path('dataSourcing/databaseConfig.yaml').open('r') as f:
        databaseConfig = yaml.full_load(f)
    irishCarPriceDatabase.deleteCarsIrelandScrappedData(
        '2022-01-02', databaseConfig)


def get_ids():
    cleanedData = list(Path(
        'C:\\Users\\Kenneth\\Documents\\Projects\\Misc\\irish_car_prices\\cleaned_data').glob('*.txt'))
    ids = []
    for file in cleanedData:
        id = file.stem[4:]
        ids.append(id)
    with open('ids.txt', 'w') as f:
        f.write(json.dumps(ids))


if __name__ == '__main__':
    # makeCarsIrelandCleanData()
    # with Path('dataSourcing/databaseConfig.yaml').open('r') as f:
    #     databaseConfig = yaml.full_load(f)
    # testConfig = databaseConfig['testConfig']

    # queriedData = irishCarPriceDatabase.selectCarsIrelandCleanData(
    #     testConfig['downloadDate'], databaseConfig)

    # for id in queriedData:
    #     for column in queriedData[id]:
    #         print(queriedData[id][column], type(queriedData[id][column]))
    deleteData()
