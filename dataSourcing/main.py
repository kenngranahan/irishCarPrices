import requests
import rawDataParser
import dataProcesser
import irishCarPriceDatabase
import yaml
from tqdm import tqdm
from datetime import datetime
from datetime import date
import json
import cloudscraper
import random
import time


def runCarsIrelandScrapper():

    def _getCurrentTimeStamp():
        return datetime.now().isoformat()

    def _getCurrentDate():
        return date.today().isoformat()
    scriptStartTS = _getCurrentTimeStamp()

    cloudscraperConfig = None
    parserConfig = None
    databaseConfig = None
    with open('dataSourcing/parserConfig.yaml', 'r') as f:
        parserConfig = yaml.full_load(f)
    with open('dataSourcing/databaseConfig.yaml', 'r') as f:
        databaseConfig = yaml.full_load(f)
    with open('dataSourcing/cloudscraperConfig.yaml', 'r') as f:
        cloudscraperConfig = yaml.full_load(f)
    with open('dataSourcing/ids.txt', 'r') as f:
        idSearchSpace = json.loads(f.read())

    idSearchSpace = idSearchSpace[:10]
    # idSearchSpace = range(2956624, 2973791)
    # idSearchSpace = irishCarPriceDatabase.selectCarsIrelandLatestId(databaseConfig)

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

    for pageId in tqdm(idSearchSpace):
        url = 'https://www.carsireland.ie/'+str(pageId)

        requestStartTS = _getCurrentTimeStamp()
        htmlSrcCode = scraper.get(url).text
        requestEndTS = _getCurrentTimeStamp()
        timeToWaitForNextRequest = random.choice(rangeOfTimesToWait)

        hitFirewall = False
        pageExists = True
        if 'Access denied' in htmlSrcCode:
            hitFirewall = True
        elif 'Page not found - CarsIreland.ie' in htmlSrcCode:
            pageExists = False

        scraperMetaData['script_start_ts'] = scriptStartTS
        scraperMetaData['page_id'] = pageId
        scraperMetaData['request_start_ts'] = requestStartTS
        scraperMetaData['request_end_ts'] = requestEndTS
        scraperMetaData['time_to_wait_for_next_request'] = timeToWaitForNextRequest
        scraperMetaData['hit_firewall'] = hitFirewall
        scraperMetaData['page_exists'] = pageExists
        for element in browser:
            scraperMetaData[element] = browser[element]
        irishCarPriceDatabase.insertCarsIrelandScraperMetaData(
            scraperMetaData, databaseConfig)

        with open('page.html', 'w') as f:
            f.write(htmlSrcCode)
        if (hitFirewall is False) and (pageExists is True):
            extractedData = rawDataParser.parseHtmlSrcCode(
                htmlSrcCode, parserConfig)
            if rawDataParser.checkParsedData(extractedData):
                extractedData = rawDataParser.confirmDataAttr(
                    extractedData, parserConfig)
                extractedData['download_date'] = _getCurrentDate()
                extractedData['id'] = pageId
                irishCarPriceDatabase.insertCarsIrelandScrappedData(
                    extractedData, databaseConfig)
        time.sleep(int(timeToWaitForNextRequest/100))


if __name__ == '__main__':
    runCarsIrelandScrapper()
