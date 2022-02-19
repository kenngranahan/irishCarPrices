import pandas as pd
import json
from pathlib import Path
import numpy as np


def _formatMake(vehicle):
    if vehicle is None:
        return None
    vehicle = vehicle.split()[0]
    return vehicle.capitalize()


def _formatModel(vehicle):
    if vehicle is None:
        return None
    vehicle = vehicle.split()[1:]
    model = ''
    for word in vehicle:
        model = model + word + ' '
    model.capitalize()
    return model.strip()


def _formatDealer(dealer):
    if dealer is None:
        return None
    return dealer.capitalize()


def _formatOwners(owners):
    if owners is None:
        return np.nan
    owners = int(owners[:-6])
    if owners == -1:
        owners = 1
    return owners


def _formatMpg(mpg):
    if mpg is None:
        return np.nan
    mpg = mpg.replace('mpg', '')
    mpg = mpg.strip()
    return int(mpg)


def _formatDoors(doors):
    if doors is None:
        return np.nan
    return int(doors[0])


def _formatYear(vehicle_year):
    if vehicle_year is None:
        return np.nan
    return int(vehicle_year)


def _formatLastUpdated(last_updated):
    if last_updated is None:
        return None
    last_updated = last_updated[8:]
    return last_updated


def _formatPrice(price):
    if price == 'P.O.A' or price is None:
        return np.nan
    price = price.replace(',', '')
    return int(price)


def _formatEngine(engine):
    if engine is None:
        return np.nan
    if engine == 'under 1.0L':
        return 0.0
    engine = engine.replace('L', '')
    engine = engine.strip()
    return float(engine)


def _formatMileage(mileage):
    if mileage is None:
        return np.nan
    mileage = mileage.replace(',', '')
    mileage = mileage.replace('km', '')
    mileage = mileage.strip()
    return int(mileage)


def _formatTax(tax):
    if tax is None:
        return np.nan
    tax = tax.replace(',', '')
    return int(tax)


def _formatNctMonth(nct):
    if nct is None:
        return None
    nct = nct[3:-5]
    nct = nct.strip()
    return str(nct)


def _formatNctYear(nct):
    if nct is None:
        return np.nan
    nct = nct[-4:]
    return int(nct)


def _formatLocation(county):
    if county is None:
        return None
    return county.capitalize()


def _formatBody(body):
    if body is None:
        return None
    return body.capitalize()


def _formatFuel(fuel):
    if fuel is None:
        return None
    return fuel.capitalize()


def _formatColor(color):
    if color is None:
        return None
    return color.capitalize()


def _formatTransmission(transmission):
    if transmission is None:
        return None
    return transmission.capitalize()


def _formatDownloadDate(downloadDate):
    if downloadDate is None:
        return None
    return downloadDate


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
                   'nct_year': ('nct', _formatNctYear, 'float'),
                   'download_date': ('download_date', _formatDownloadDate, 'O')}


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
