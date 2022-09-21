# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 16:01:42 2022

@author: User

"""

from tkinter import *
from tkinter import ttk
from ModelAPI import carPricePredictor

CFG = {
       'entryWidth': 7
       }
priceModel = carPricePredictor()
root = Tk()

root.title("Feet to Meters")
mainframe = ttk.Frame(root, padding="3 3 12 12")
mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)


variables = {}
entries = {}
expectedColumns = priceModel.getExpectedColumns()
for column in expectedColumns:
    entries[column] = ttk.Entry(mainframe, CFG['entryWidth'])
    if expectedColumns[column] == 'object':
        variables[column] = StringVar()
    else:
        variables[column] = DoubleVar()
        
