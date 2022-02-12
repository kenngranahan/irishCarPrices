import re


def parseHtmlSrcCode(htmlSrcCode, scrapperConfig):
    config = scrapperConfig['parseHtmlSrcCode']
    htmlSrcCode = re.sub('>\s*<', '><', htmlSrcCode)

    def keepTextBtwnTags(htmlText, startTag, endTag, keepStartTag=True, keepEndTag=True):
        startIdx = htmlText.find(startTag)
        endIdx = htmlText[startIdx+len(startTag):].find(endTag)
        endIdx += startIdx+len(startTag)
        if not keepStartTag:
            startIdx += len(startTag)
        if keepEndTag:
            endIdx += len(endTag)
        return htmlText[:endIdx][startIdx:]

    def removeTextBtwnTags(htmlText, startTag, endTag, removeStartTag=True, removeEndTag=True):
        startIdx = htmlText.find(startTag)
        endIdx = htmlText[startIdx+len(startTag):].find(endTag)
        endIdx += startIdx+len(startTag)
        if not removeStartTag:
            startIdx += len(startTag)
        if removeEndTag:
            endIdx += len(endTag)
        return htmlText[:startIdx] + htmlText[endIdx:]
    dataExtractedFromHtml = {}
    for parameter in config:
        resultOfAction = ''
        startTag = parameter[0]
        endTag = parameter[1]
        action = parameter[2]
        includesStartTag = parameter[3]
        includesEndTag = parameter[4]
        variableName = parameter[5]

        if action == 'keep':
            resultOfAction = keepTextBtwnTags(htmlSrcCode,
                                              startTag=startTag,
                                              endTag=endTag,
                                              keepStartTag=includesStartTag,
                                              keepEndTag=includesEndTag)
        elif action == 'remove':
            resultOfAction = removeTextBtwnTags(htmlSrcCode,
                                                startTag=startTag,
                                                endTag=endTag,
                                                removeStartTag=includesStartTag,
                                                removeEndTag=includesEndTag)
        if variableName == '':
            htmlSrcCode = resultOfAction
        else:
            resultOfAction = resultOfAction.strip()
            resultOfAction = resultOfAction.encode('ascii', errors='ignore')
            resultOfAction = resultOfAction.decode('ascii', errors='ignore')
            dataExtractedFromHtml[variableName] = resultOfAction

    return dataExtractedFromHtml


def checkParsedData(extractedData):
    if 'dealer' not in extractedData:
        return False
    elif extractedData['dealer'] == '':
        return False
    return True


def confirmDataAttr(extractedData, scrapperConfig):

    config = scrapperConfig['confirmDataAttr']
    transmissions = config['transmission']
    body = config['body']
    fuel = config['fuel']
    color = config['color']

    def is_transmission(attrValue):
        return (attrValue in transmissions)

    def is_engine(attrValue):
        return (attrValue[-1] == 'L' or attrValue == 'Unknown')

    def is_body(attrValue):
        return (attrValue in body)

    def is_fuel(attrValue):
        return (attrValue in fuel)

    def is_doors(attrValue):
        return (len(attrValue) > 5 and (attrValue[-5:] == 'Doors' or attrValue[-4:] == 'Door'))

    def is_mpg(attrValue):
        return (len(attrValue) > 3 and attrValue[-3:] == 'mpg')

    def is_owner(attrValue):
        return (len(attrValue) >= 7 and (attrValue[-6:] == 'Owners' or attrValue[-5:] == 'Owner'))

    def is_color(attrValue):
        return (attrValue in color)

    def is_tax(attrValue):
        return (len(attrValue) > 0 and attrValue.isnumeric())

    def is_nct(attrValue):
        return (len(attrValue) > 3 and attrValue[:3] == 'NCT')

    confirmAttr = {'transmission': is_transmission,
                   'engine': is_engine,
                   'body': is_body,
                   'fuel': is_fuel,
                   'doors': is_doors,
                   'mpg': is_mpg,
                   'owners': is_owner,
                   'color': is_color,
                   'tax': is_tax,
                   'nct': is_nct}
    correctAttr = {}

    for attr in confirmAttr:
        attrValue = extractedData[attr]
        if attrValue != '':
            attrIsCorrect = confirmAttr[attr](attrValue)

            if not attrIsCorrect:
                for otherAttr in confirmAttr:
                    otherAttrIsCorrect = confirmAttr[otherAttr](attrValue)
                    if otherAttrIsCorrect:
                        correctAttr[otherAttr] = attrValue
                extractedData[attr] = ''

    if len(correctAttr) != 0:
        for attr in correctAttr:
            extractedData[attr] = correctAttr[attr]

    return extractedData
