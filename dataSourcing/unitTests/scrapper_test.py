import unittest
import json
import yaml
import rawDataParser
import dataProcesser


class rawDataScrapperTestCase(unittest.TestCase):

    def test_parser(self):
        with open('dataSourcing/scrapperConfig.yaml', 'r') as f:
            scrapperConfig = yaml.full_load(f)
        with open(scrapperConfig['testConfig']['htmlParser'] + 'input.txt', 'r') as f:
            extractedData = f.read()
        with open(scrapperConfig['testConfig']['htmlParser'] + 'output.txt', 'r') as f:
            testOutput = json.loads(f.read())

        extractedData = rawDataParser.parseHtmlSrcCode(
            extractedData, scrapperConfig['parseHtmlSrcCode'])

        for attr in testOutput:
            expectedOutcome = testOutput[attr]
            actualOutcome = extractedData[attr]
            self.assertEqual(expectedOutcome, actualOutcome, attr)

    def test_confirmDataAttr(self):
        with open('dataSourcing/scrapperConfig.yaml', 'r') as f:
            scrapperConfig = yaml.full_load(f)
        with open(scrapperConfig['testConfig']['htmlParser'] + 'input.txt', 'r') as f:
            extractedData = f.read()
        with open(scrapperConfig['testConfig']['htmlParser'] + 'output.txt', 'r') as f:
            testOutput = json.loads(f.read())

        extractedData = rawDataParser.parseHtmlSrcCode(
            extractedData, scrapperConfig['parseHtmlSrcCode'])
        extractedData = rawDataParser.confirmDataAttr(
            extractedData, scrapperConfig['confirmDataAttr'])

        for attr in testOutput:
            expectedOutcome = testOutput[attr]
            actualOutcome = extractedData[attr]
            self.assertEqual(expectedOutcome, actualOutcome, attr)

    def test_1646849(self):
        with open('dataSourcing/scrapperConfig.yaml', 'r') as f:
            scrapperConfig = yaml.full_load(f)
        with open(scrapperConfig['testConfig']['1646849'] + 'input.txt', 'r') as f:
            extractedData = f.read()
        with open(scrapperConfig['testConfig']['1646849'] + 'output.txt', 'r') as f:
            testOutput = json.loads(f.read())

        extractedData = rawDataParser.parseHtmlSrcCode(
            extractedData, scrapperConfig['parseHtmlSrcCode'])
        successfulParse = rawDataParser.checkParsedData(extractedData)
        self.assertTrue(successfulParse)

        extractedData['download_date'] = '2021-12-30'
        extractedData['id'] = '1646849'
        extractedData = rawDataParser.confirmDataAttr(
            extractedData, scrapperConfig['confirmDataAttr'])

        self.maxDiff = None
        self.assertDictEqual(extractedData, testOutput)

    def test_2973707(self):
        with open('dataSourcing/scrapperConfig.yaml', 'r') as f:
            scrapperConfig = yaml.full_load(f)

        with open(scrapperConfig['testConfig']['2973707'] + 'input.txt', 'r') as f:
            extractedData = f.read()
        with open(scrapperConfig['testConfig']['2973707'] + 'output.txt', 'r') as f:
            testOutput = json.loads(f.read())

        extractedData = rawDataParser.parseHtmlSrcCode(
            extractedData, scrapperConfig['parseHtmlSrcCode'])
        successfulParse = rawDataParser.checkParsedData(extractedData)
        self.assertTrue(successfulParse)

        extractedData['download_date'] = '2021-12-30'
        extractedData['id'] = '2973707'
        extractedData = rawDataParser.confirmDataAttr(
            extractedData, scrapperConfig['confirmDataAttr'])
        self.assertDictEqual(extractedData, testOutput)

    def test_2973737(self):
        with open('dataSourcing/scrapperConfig.yaml', 'r') as f:
            scrapperConfig = yaml.full_load(f)

        with open(scrapperConfig['testConfig']['2973737'] + 'input.txt', 'r') as f:
            extractedData = f.read()
        extractedData = rawDataParser.parseHtmlSrcCode(
            extractedData, scrapperConfig['parseHtmlSrcCode'])
        successfulParse = rawDataParser.checkParsedData(extractedData)
        self.assertFalse(successfulParse)

    def test_2973782(self):
        with open('dataSourcing/scrapperConfig.yaml', 'r') as f:
            scrapperConfig = yaml.full_load(f)
        with open(scrapperConfig['testConfig']['2973782'] + 'input.txt', 'r') as f:
            extractedData = f.read()
        with open(scrapperConfig['testConfig']['2973782'] + 'output.txt', 'r') as f:
            testOutput = json.loads(f.read())

        extractedData = rawDataParser.parseHtmlSrcCode(
            extractedData, scrapperConfig['parseHtmlSrcCode'])
        successfulParse = rawDataParser.checkParsedData(extractedData)
        self.assertTrue(successfulParse)
        extractedData['download_date'] = '2021-12-30'
        extractedData['id'] = '2973782'
        extractedData = rawDataParser.confirmDataAttr(
            extractedData, scrapperConfig['confirmDataAttr'])
        self.assertDictEqual(extractedData, testOutput)

    def test_2979515(self):
        with open('dataSourcing/scrapperConfig.yaml', 'r') as f:
            scrapperConfig = yaml.full_load(f)
        with open(scrapperConfig['testConfig']['2979515'] + 'input.txt', 'r') as f:
            extractedData = f.read()
        with open(scrapperConfig['testConfig']['2979515'] + 'output.txt', 'r') as f:
            testOutput = json.loads(f.read())

        extractedData = rawDataParser.parseHtmlSrcCode(
            extractedData, scrapperConfig['parseHtmlSrcCode'])
        successfulParse = rawDataParser.checkParsedData(extractedData)
        self.assertTrue(successfulParse)

        extractedData['download_date'] = '2021-12-30'
        extractedData['id'] = '2979515'
        extractedData = rawDataParser.confirmDataAttr(
            extractedData, scrapperConfig['confirmDataAttr'])
        self.assertDictEqual(extractedData, testOutput)
