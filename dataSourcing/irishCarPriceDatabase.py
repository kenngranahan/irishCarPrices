import numpy as np
import psycopg2


def _connectToDatabase(databaseConfig):
    DSN = databaseConfig['DSN']
    conn = psycopg2.connect(dbname=DSN['dbname'],
                            user=DSN['user'],
                            password=DSN['password'],
                            host=DSN['host'],
                            port=DSN['port'],
                            connect_timeout=DSN['connect_timeout'])
    return conn


def insertCarsIrelandScrappedData(scrappedData, databaseConfig):
    conn = _connectToDatabase(databaseConfig)

    columnsToInsert = ''
    valuesToInsert = ''
    for columnName, columnType in databaseConfig['carsIrelandScrappedDataSchema']:
        value = scrappedData[columnName]

        if columnsToInsert == '':
            columnsToInsert = columnName
            if value == "":
                valuesToInsert = 'NULL'
            elif columnType == 'numeric':
                valuesToInsert = str(value)
            else:
                valuesToInsert = '\''+str(value)+'\''
        else:
            columnsToInsert = columnsToInsert + ','+columnName
            if value == "":
                valuesToInsert = valuesToInsert + ',NULL'
            elif columnType == 'numeric':
                valuesToInsert = valuesToInsert + ','+str(value)
            else:
                valuesToInsert = valuesToInsert + ',\''+str(value)+'\''

    with conn:
        with conn.cursor() as cursor:
            query = 'INSERT INTO public."carsIrelandScrappedData"(' + \
                columnsToInsert+') VALUES ('+valuesToInsert+');'
            cursor.execute(query)
            conn.commit()
        return None


def selectCarsIrelandScrappedData(downloadDate, databaseConfig):
    conn = _connectToDatabase(databaseConfig)
    with conn:
        with conn.cursor() as cursor:
            query = 'SELECT * FROM public."carsIrelandScrappedData" WHERE download_date = \'' + \
                str(downloadDate)+'\';'
            cursor.execute(query)
            queriedData = {}
            id = None
            for record in cursor:
                recordData = {}
                for valueQueried, (columnName, _) in zip(record, databaseConfig['carsIrelandScrappedDataSchema']):
                    recordData[columnName] = valueQueried
                    if columnName == 'id':
                        id = valueQueried
                queriedData[id] = recordData
            return queriedData


def deleteCarsIrelandScrappedData(downloadDate, databaseConfig):
    conn = _connectToDatabase(databaseConfig)
    with conn:
        with conn.cursor() as cursor:
            query = 'DELETE FROM public."carsIrelandScrappedData" WHERE download_date = \'' + \
                str(downloadDate)+'\';'
            cursor.execute(query)
            return None


def insertCarsIrelandCleanData(cleanData, databaseConfig):
    conn = _connectToDatabase(databaseConfig)

    columnsToInsert = ''
    valuesToInsert = ''
    for columnName, columnType in databaseConfig['carsIrelandCleanDataSchema']:
        value = cleanData[columnName]

        if columnsToInsert == '':
            columnsToInsert = columnName
            if isinstance(value, object):
                valuesToInsert = '\''+str(value)+'\''
            elif isinstance(value, float):
                if np.isnan(value):
                    valuesToInsert = 'NULL'
                else:
                    valuesToInsert = str(value)
        else:
            columnsToInsert = columnsToInsert + ','+columnName
            if isinstance(value, object):
                valuesToInsert = valuesToInsert + ',\''+str(value)+'\''
            elif isinstance(value, float):
                if np.isnan(value):
                    valuesToInsert = valuesToInsert + ',NULL'
                else:
                    valuesToInsert = valuesToInsert + ','+str(value)

    with conn:
        with conn.cursor() as cursor:
            query = 'INSERT INTO public."carsIrelandCleanData"(' + \
                columnsToInsert+') VALUES ('+valuesToInsert+');'
            cursor.execute(query)
            conn.commit()
        return None


def selectCarsIrelandCleanData(downloadDate, databaseConfig):
    conn = _connectToDatabase(databaseConfig)
    with conn:
        with conn.cursor() as cursor:
            query = 'SELECT * FROM public."carsIrelandCleanData" WHERE download_date = \'' + \
                downloadDate+'\';'
            cursor.execute(query)
            queriedData = {}
            id = None
            for record in cursor:
                recordData = {}
                for valueQueried, (columnName, columnType) in zip(record, databaseConfig['carsIrelandCleanDataSchema']):
                    if valueQueried == 'nan':
                        if columnType == 'numeric':
                            recordData[columnName] = np.nan
                        else:
                            recordData[columnName] = ''
                    elif columnType == 'numeric':
                        recordData[columnName] = float(valueQueried)
                    else:
                        recordData[columnName] = valueQueried

                    if columnName == 'id':
                        id = valueQueried
                queriedData[id] = recordData
            return queriedData


def deleteCarsIrelandCleanData(downloadDate, databaseConfig):
    conn = _connectToDatabase(databaseConfig)
    with conn:
        with conn.cursor() as cursor:
            query = 'DELETE FROM public."carsIrelandCleanData" WHERE download_date = \'' + \
                str(downloadDate)+'\';'
            cursor.execute(query)
            return None


def selectCarsIrelandLatestId(databaseConfig):
    conn = _connectToDatabase(databaseConfig)
    query = 'SELECT Id FROM public."carsIrelandCleanData" WHERE download_date = MAX(download_date)'
    with conn.cursor() as cursor:
        ids = cursor.execute(query)
    return ids
