import pandas as pd
import json
from pathlib import Path
import numpy as np


def _formatMake(vehicle):
    vehicle = vehicle.split()[0]
    return vehicle.capitalize()


def _formatModel(vehicle):
    vehicle = vehicle.split()[1:]
    model = ''
    for word in vehicle:
        model = model + word + ' '
    model.capitalize()
    return model.strip()


def _formatDealer(dealer):
    return dealer.capitalize()


def _formatOwners(owners):
    if owners == '':
        return np.nan
    owners = int(owners[:-6])
    if owners == -1:
        owners = 1
    return owners


def _formatMpg(mpg):
    if mpg == '':
        return np.nan
    mpg = mpg.replace('mpg', '')
    mpg = mpg.strip()
    return int(mpg)


def _formatDoors(doors):
    if doors == '':
        return np.nan
    return int(doors[0])


def _formatYear(vehicle_year):
    if vehicle_year == '':
        return np.nan
    return int(vehicle_year)


def _formatLastUpdated(last_updated):
    if last_updated == '':
        return None
    last_updated = last_updated[8:]
    return last_updated


def _formatPrice(price):
    if price == 'P.O.A':
        return np.nan
    price = price.replace(',', '')
    return int(price)


def _formatEngine(engine):
    if engine == '':
        return np.nan
    engine = engine.replace('L', '')
    engine = engine.strip()
    return float(engine)


def _formatMileage(mileage):
    if mileage == '':
        return np.nan
    mileage = mileage.replace(',', '')
    mileage = mileage.replace('km', '')
    mileage = mileage.strip()
    return int(mileage)


def _formatTax(tax):
    if tax == '':
        return np.nan
    tax = tax.replace(',', '')
    return int(tax)


def _formatNctMonth(nct):
    if nct == '':
        return None
    nct = nct[3:-5]
    nct = nct.strip()
    return str(nct)


def _formatNctYear(nct):
    if nct == '':
        return np.nan
    nct = nct[-4:]
    return int(nct)


def _formatLocation(county):
    return county.capitalize()


def _formatBody(body):
    return body.capitalize()


def _formatFuel(fuel):
    return fuel.capitalize()


def _formatColor(color):
    return color.capitalize()


def _formatTransmission(transmission):
    return transmission.capitalize()


attr_formatting = {'dealer': ('dealer', _formatDealer, 'O'),
                   'county': ('county', _formatLocation, 'O'),
                   'make': ('vehicle', _formatMake, 'O'),
                   'model': ('vehicle', _formatModel, 'O'),
                   'vehicle_year': ('vehicle_year', _formatYear, 'float'),
                   'mileage': ('mileage', _formatMileage, 'float'),
                   'last_updated': ('last_updated', _formatLastUpdated, 'O'),
                   'price': ('price', _formatPrice, 'float'),
                   'transmission': ('transmission', _formatTransmission, 'O'),
                   'engine': ('engine', _formatEngine, 'float'),
                   'body': ('body', _formatBody, 'O'),
                   'fuel': ('fuel', _formatFuel, 'O'),
                   'doors': ('doors', _formatDoors, 'float'),
                   'mpg': ('mpg', _formatMpg, 'float'),
                   'owners': ('owners', _formatOwners, 'float'),
                   'color': ('color', _formatColor, 'O'),
                   'tax': ('tax', _formatTax, 'float'),
                   'nct_month': ('nct', _formatNctMonth, 'O'),
                   'nct_year': ('nct', _formatNctYear, 'float')}


def _aggregateData(scrappedDataFolder):
    scrappedDataFiles = list(Path(scrappedDataFolder).glob('*.txt'))
    aggregatedData = None
    for dataFile in scrappedDataFiles:
        with open(dataFile, 'r') as f:
            scrappedData = json.loads(f.read())
        fileId = dataFile.stem
        if aggregatedData is None:
            aggregatedData = pd.DataFrame(scrappedData, index=[fileId])
        else:
            aggregatedData = aggregatedData.append(
                pd.DataFrame(scrappedData, index=[fileId]))
    return aggregatedData


def _formatData(aggregatedData):
    for _formattedAttrName in attr_formatting:
        attr, attr_formatter, _formattedAttrDtype = attr_formatting[_formattedAttrName]
        aggregatedData[_formattedAttrName] = aggregatedData[attr].apply(
            lambda data: attr_formatter(data))
        aggregatedData = aggregatedData.astype(
            {_formattedAttrName: _formattedAttrDtype}, errors='ignore')

    for column in aggregatedData.columns:
        if column not in attr_formatting:
            aggregatedData.drop(column, axis=1, inplace=True)
    return aggregatedData


def cleanRawData(rawData):
    cleanData = pd.DataFrame().from_dict(rawData, orient='index')
    cleanData = _formatData(cleanData)
    return cleanData.to_dict(orient='index')
