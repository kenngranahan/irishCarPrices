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


def _makeInsertQuery(dataDict, databaseConfig, tableName):
    columnsToInsert = ''
    valuesToInsert = ''
    for columnName, _ in databaseConfig[tableName]:
        value = dataDict[columnName]

        if columnsToInsert == '':
            columnsToInsert = columnName
            if isinstance(value, object):
                if value == '':
                    valuesToInsert = 'NULL'
                else:
                    valuesToInsert = '\''+str(value)+'\''
            elif isinstance(value, float):
                if np.isnan(value):
                    valuesToInsert = 'NULL'
                else:
                    valuesToInsert = str(value)
            elif isinstance(value, bool):
                if value is True:
                    valuesToInsert = '1'
                elif value is False:
                    valuesToInsert = '0'
                else:
                    valuesToInsert = 'NULL'
            elif value is None:
                valuesToInsert = 'NULL'
        else:
            columnsToInsert = columnsToInsert + ','+columnName
            if isinstance(value, object):
                if value == '':
                    valuesToInsert = valuesToInsert + ',NULL'
                else:
                    valuesToInsert = valuesToInsert + ',\''+str(value)+'\''
            elif isinstance(value, float):
                if np.isnan(value):
                    valuesToInsert = valuesToInsert + ',NULL'
                else:
                    valuesToInsert = valuesToInsert + ','+str(value)
            elif isinstance(value, bool):
                if value is True:
                    valuesToInsert = valuesToInsert + ',1'
                elif value is False:
                    valuesToInsert = valuesToInsert + ',0'
                else:
                    valuesToInsert = valuesToInsert + ',NULL'
            elif value is None:
                valuesToInsert = valuesToInsert + ',NULL'

    query = 'INSERT INTO public."' + tableName + '"(' + \
        columnsToInsert+') VALUES ('+valuesToInsert+');'
    return query


def insertCarsIrelandScrappedData(scrappedData, databaseConfig):
    conn = _connectToDatabase(databaseConfig)
    insertQuery = _makeInsertQuery(
        scrappedData, databaseConfig, 'carsIrelandScrappedData')
    with conn:
        with conn.cursor() as cursor:
            cursor.execute(insertQuery)
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
                for valueQueried, (columnName, _) in zip(record, databaseConfig['carsIrelandScrappedData']):
                    recordData[columnName] = valueQueried
                    if columnName == 'page_id':
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
    with conn:
        for pageId in cleanData:
            pageData = cleanData[pageId]
            pageData['page_id'] = pageId
            insertQuery = _makeInsertQuery(
                pageData, databaseConfig, 'carsIrelandCleanData')

            with conn.cursor() as cursor:
                cursor.execute(insertQuery)
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
                for valueQueried, (columnName, columnType) in zip(record, databaseConfig['carsIrelandCleanData']):
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


def selectCarsIrelandRecentIds(databaseConfig, startDate, endDate):
    conn = _connectToDatabase(databaseConfig)
    query = 'SELECT page_id FROM public."carsIrelandScrappedData" \
             WHERE page_exists = True \
             AND script_start_ts BETWEEN \'' + startDate + '\' AND \'' + endDate + '\';'
    with conn.cursor() as cursor:
        ids = cursor.execute(query)
    return ids


def insertCarsIrelandScraperMetaData(scraperMetaData, databaseConfig):
    conn = _connectToDatabase(databaseConfig)
    insertQuery = _makeInsertQuery(
        scraperMetaData, databaseConfig, 'carsIrelandScraperMetaData')
    with conn:
        with conn.cursor() as cursor:
            cursor.execute(insertQuery)
            conn.commit()
        return None
