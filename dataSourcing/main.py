import requests
import rawDataParser
import dataProcesser
import irishCarPriceDatabase
import yaml
from tqdm import tqdm
from datetime import datetime
import json
import cloudscraper
import random
import time


def runCarsIrelandScrapper():

    def getCurrentTimeStamp():
        return datetime.now().replace(microsecond=0).isoformat()
    scriptStartTS = getCurrentTimeStamp()

    cloudscraperConfig = None
    parserConfig = None
    databaseConfig = None
    with open('parserConfig.yaml', 'r') as f:
        parserConfig = yaml.full_load(f)
    with open('databaseConfig.yaml', 'r') as f:
        databaseConfig = yaml.full_load(f)
    with open('cloudscraperConfig.yaml', 'r') as f:
        cloudscraperConfig = yaml.full_load(f)

    successfulFilters = cloudscraperConfig['successfulFilters']
    minimumCentisecondsBetweenRequests = cloudscraperConfig['minimumCentisecondsBetweenRequests']
    maximumCentiecondsBetweenRequests = cloudscraperConfig['maximumCentiecondsBetweenRequests']
    timeIncrement = cloudscraperConfig['timeIncrement']
    rangeOfTimesToWait = range(
        minimumCentisecondsBetweenRequests, maximumCentiecondsBetweenRequests, timeIncrement)

    browser = random.choice(
        successfulFilters[cloudscraper.__version__])
    scraper = cloudscraper.create_scraper(browser=browser)
    scraperMetaData = {}

    with open('ids.txt', 'r') as f:
        idSearchSpace = json.loads(f.read())
    # idSearchSpace = range(2956624, 2973791)
    # idSearchSpace = irishCarPriceDatabase.selectCarsIrelandLatestId(databaseConfig)
    idSearchSpace = idSearchSpace[:5]

    for pageId in tqdm(idSearchSpace):
        url = 'https://www.carsireland.ie/'+str(pageId)

        requestStartTS = getCurrentTimeStamp()
        htmlSrcCode = scraper.get(url).text
        requestEndTS = getCurrentTimeStamp()
        timeToWaitForNextRequest = random.choice(rangeOfTimesToWait)

        scraperMetaData['scriptStartTS'] = scriptStartTS
        scraperMetaData['pageId'] = pageId
        scraperMetaData['requestStartTS'] = requestStartTS
        scraperMetaData['requestEndTS'] = requestEndTS
        scraperMetaData['timeToWaitForNextRequest'] = timeToWaitForNextRequest
        for element in browser:
            scraperMetaData[element] = browser[element]

        with open(pageId, 'w') as f:
            f.write(json.dumps(htmlSrcCode))
        htmlSrcCode = rawDataParser.parseHtmlSrcCode(
            htmlSrcCode, parserConfig)

        if False:
            if rawDataParser.checkParsedData(htmlSrcCode):
                htmlSrcCode = rawDataParser.confirmDataAttr(
                    htmlSrcCode, parserConfig)
                htmlSrcCode['download_date'] = downloadDate
                htmlSrcCode['id'] = pageId
                irishCarPriceDatabase.insertCarsIrelandScrappedData(
                    htmlSrcCode, databaseConfig)
        time.sleep(int(timeToWaitForNextRequest/100))
    # rawData = irishCarPriceDatabase.selectCarsIrelandScrappedData(
    #     downloadDate, databaseConfig)
    # cleanData = dataProcesser.cleanRawData(rawData, databaseConfig)
    # irishCarPriceDatabase.insertCarsIrelandCleanData(cleanData, databaseConfig)


if __name__ == '__main__':
    runCarsIrelandScrapper()
