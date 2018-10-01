#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 21:07:20 2018

@author: Valenty
"""
#This small script was used to manipulate the many datasets I had
#This script basically opens each csv file I had and creates a new csv file
#The new csv file has 30 x,y,z plots per row, instead of the previous 1 per row

from itertools import zip_longest
import csv

#I changed these input and outputs depending on which data file I was...
#... manipulating

f_input = open('non-fall/ANALOG26.txt', newline='')
f_output = open('non-fall/ANALOG26out.csv', 'w', newline='')

with f_input, f_output:
    csv_input = csv.reader(f_input)
    csv_output = csv.writer(f_output)

    for rows in zip_longest(*[iter(csv_input)] * 30, fillvalue=None):
        rows = [[float(row[0])] + row[1:] for row in rows if row]
        delta = rows[-1][0] - rows[0][0]
        combined = [delta]
        for row in rows:
            combined.extend([row[1], row[2], row[3]])
            
    
        csv_output.writerow(combined)
    
    
    