import requests
import rawDataParser
import dataProcesser
import irishCarPriceDatabase
import yaml
from tqdm import tqdm
from datetime import datetime, timedelta
from datetime import date
import json
import cloudscraper
import random
import time


def runCarsIrelandScrapper(idSearchSpace, cloudscraperConfig, parserConfig, databaseConfig):

    def _getCurrentTimeStamp():
        return datetime.now().isoformat()

    def _getCurrentDate():
        return date.today().isoformat()
    scriptStartTS = _getCurrentTimeStamp()

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

    for pageId in idSearchSpace:
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

        if (hitFirewall is False) and (pageExists is True):
            extractedData = rawDataParser.parseHtmlSrcCode(
                htmlSrcCode, parserConfig)
            if rawDataParser.checkParsedData(extractedData):
                extractedData = rawDataParser.confirmDataAttr(
                    extractedData, parserConfig)
                extractedData['download_date'] = _getCurrentDate()
                extractedData['page_id'] = pageId
                irishCarPriceDatabase.insertCarsIrelandScrappedData(
                    extractedData, databaseConfig)
        time.sleep(int(timeToWaitForNextRequest/100))
    scraper.close()


def runCarsIrelandCleaner(downloadDate, databaseConfig):
    scrappedData = irishCarPriceDatabase.selectCarsIrelandScrappedData(
        downloadDate, databaseConfig)
    if scrappedData != {}:
        cleanData = dataProcesser.cleanRawData(scrappedData)
        irishCarPriceDatabase.insertCarsIrelandCleanData(
            cleanData, databaseConfig)
    return None


if __name__ == '__main__':

    cloudscraperConfig = None
    parserConfig = None
    databaseConfig = None
    with open('dataSourcing/configs/parserConfig.yaml', 'r') as f:
        parserConfig = yaml.full_load(f)
    with open('dataSourcing/configs/databaseConfig.yaml', 'r') as f:
        databaseConfig = yaml.full_load(f)
    with open('dataSourcing/configs/cloudscraperConfig.yaml', 'r') as f:
        cloudscraperConfig = yaml.full_load(f)

    # Weeks are assumed to start on Monday and end on Sunday
    currentDate = date.today()
    thisWeekStartDate = currentDate - timedelta(days=currentDate.weekday())
    thisWeekEndDate = currentDate + timedelta(days=(6-currentDate.weekday()))

    lastWeekStartDate = currentDate - timedelta(days=(7+currentDate.weekday()))
    lastWeekEndDate = currentDate - timedelta(days=(1+currentDate.weekday()))

    idsToSearch = irishCarPriceDatabase.selectCarsIrelandIdsDownloaded(
        databaseConfig, lastWeekStartDate, lastWeekEndDate + timedelta(days=1))
    if idsToSearch != []:
        idsAddedToWebsite = [
            max(idsToSearch)+(idx+1) for idx in range(cloudscraperConfig['idsAddedWeekly'])]
        idsToSearch.extend(idsAddedToWebsite)

        idsSearchedThisWeek = irishCarPriceDatabase.selectCarsIrelandIdsDownloaded(
            databaseConfig, thisWeekStartDate, currentDate + timedelta(days=1))
        for id in idsSearchedThisWeek:
            idsToSearch.remove(id)

        numberOfBins = int(len(idsToSearch)/cloudscraperConfig['idBinSize'])
        for idx in tqdm(range(numberOfBins-1)):
            startIdx = idx*cloudscraperConfig['idBinSize']
            endIdx = (idx+1)*cloudscraperConfig['idBinSize']
            binnedIds = idsToSearch[startIdx:endIdx]
            runCarsIrelandScrapper(
                binnedIds, cloudscraperConfig, parserConfig, databaseConfig)

    latestCleanDate = irishCarPriceDatabase.selectLatestCleanDate(
        databaseConfig)
    if latestCleanDate is None:
        latestCleanDate = date(2022, 2, 1)

    if latestCleanDate != currentDate:
        datesToRunCleaner = [
            latestCleanDate + timedelta(days=(idx+1)) for idx in range((currentDate-latestCleanDate).days)]
        for cleanDate in datesToRunCleaner:
            runCarsIrelandCleaner(cleanDate, databaseConfig)
