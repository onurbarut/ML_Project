# This code is prepared by Onur Barut for Project Assignment 
# of COMP.5450 (FALL 2018) course, UML,  by Dr. Jerome Braun.
#
# References:
# 1- https://www.kaggle.com/charma69/titanic-data-science-solutions/edit

# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd
from read_data import *

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline # IPython line, doesn't work in script

import os

directory = 'Data_Analysis_outputs'
if not os.path.exists(directory):
	os.makedirs(directory)

names=['ID', 'ClumpTkns', 'UnofCSize', 'UnofCShape', 'MargAdh', 
		'SngEpiCSize', 'BareNuc', 'BlandCrmtn', 'NrmlNuc', 'Mitoses', 'Malignant']

SAVE = False

def main():
	# 1.import data
	dataset = 'WBCD.csv'
	Data = read_data(dataset)

	# 2.Fill with median
	Data = fillMed(Data)


	# dnalyze by describing data
	#print(Data.columns.values)

	# preview the data
	#print(Data.head())
	#print(Data.tail())

	#print(Data.describe())

	#print(Data.describe(include=['O']))

	for i in range(1, (Data.shape[1]-1)):
		#print(Data[[names[i], names[-1]]].groupby([names[i]], \
		#	as_index=False).mean().sort_values(by=names[-1], ascending=False))
		g = sns.FacetGrid(Data, col=names[-1])
		g.map(plt.hist, names[i], bins=20)
		if SAVE == True:
			plt.savefig(directory+'/'+names[i]+'_'+names[-1]+'.png')
		plt.close()


	fig = plt.figure()
	ax = fig.add_subplot(111)
	cax = ax.matshow(Data.corr(), vmin=-1, vmax=1)
	fig.colorbar(cax)
	ticks = np.arange(0, Data.shape[1], 1)
	ax.set_xticks(ticks)
	ax.set_yticks(ticks)
	ax.set_xticklabels(names, fontsize = 6)
	ax.set_yticklabels(names)
	if SAVE == True:
		plt.savefig(directory+'/'+'CorrMat.png')
	plt.close()

	corrArray = Data.corr().values()[-1,:]
	corrFrame = pd.DataFrame(data=corrArray,
								index=names,
								columns=['Correlation'])
	corrFrame_sorted = corrFrame.sort_values(['Correlation'], ascending=False) 
	#print(corrFrame_sorted)

	indexBar = range(len(names)-2)
	plt.bar(indexBar, (corrFrame_sorted.values)[1:-1], align='center')
	plt.xlabel('Features', fontsize=16)
	plt.ylabel('Correlation of tumor being malignant', fontsize=16)
	plt.xticks(indexBar, (corrFrame_sorted.index.values)[1:-1], fontsize=6, rotation=0)
	plt.ylim((0,1))
	if SAVE == True:
		plt.savefig(directory+'/'+'CorrBar.png')
	plt.close()
"""	
	grid = sns.FacetGrid(Data, col=names[-1], row=names[1], height=2.2, aspect=1.6)
	grid.map(plt.hist, names[2], alpha=.5, bins=20)
	grid.add_legend();
"""


if __name__ == '__main__':
    main()
