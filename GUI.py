# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 16:01:42 2022

@author: User

"""

from tkinter import *
from tkinter import ttk

from ModelAPI import carPricePredictor
import pandas as pd

def getDataFrame(variables):
    values=[]
    columns = []
    for key, value in variables.items():
        columns.append(key)
        values.append(value.get())
        
    df = pd.DataFrame(data=[values], columns = columns)
    return df

def userInputValues(variables):
    valuesAreInputed = True
    for key in CFG['requiredInputs']:
        value = variables[key]
        if value.get() == '':
            valuesAreInputed = False
    return valuesAreInputed

def priceCar():
    if userInputValues(variables):
        X = getDataFrame(variables)
        price = priceModel.predictPrice(X)
        price = price.values[0]
        carPrice.set(price)
        



  
   
priceModel = carPricePredictor()
root = Tk()


root.title("Car Pricer")
mainframe = ttk.Frame(root, padding="3 3 12 12")
mainframe.grid(column=0, row=0)
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)


variables = {}
entries = {}
labels = {}
expectedColumns = priceModel.getExpectedColumns()
for idx, column in enumerate(expectedColumns):

    variables[column] = StringVar()
    
    
    entries[column] = ttk.Entry(mainframe, 
                                width = CFG['inputEntryWidth'],
                                textvariable=variables[column])
    entries[column].grid(column=1, row=idx)
    
    labels[column] = ttk.Label(mainframe, text=column)
    labels[column].grid(column=2, row=idx)

carPrice = StringVar()
ttk.Label(mainframe, textvariable=carPrice).grid(column=3, row=4)
ttk.Button(mainframe, text='Price my Car', command=priceCar).grid(column=3, row=3)


root.mainloop()
