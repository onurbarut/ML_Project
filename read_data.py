import pandas as pd
import numpy as np 
import os

def read_data(dataset):
	path = os.getcwd()
	dataset = path + "/" + dataset
	main_df = pd.DataFrame() # begin empty

	names=['ID', 'ClumpTkns', 'UnofCSize', 'UnofCShape', 'MargAdh', 
		'SngEpiCSize', 'BareNuc', 'BlandCrmtn', 'NrmlNuc', 'Mitoses', 'Class']
		
	df = pd.read_csv(dataset, names=names)  # read in specific file
	df['Class'] = df['Class'].replace([4, 2],[1, 0])
	colval = df.columns.values
	colvalst = colval.tolist()  
	colvalst[-1] = 'Malignant'
	df.columns = colvalst
	main_df = df
	#main_df.fillna(method="ffill", inplace=True)  # if there are gaps in data, use previously known values
	#print(main_df.head())  # how did we do??
	return main_df

# get rid of the missing examplars
def clean_data(data_df):
	data_df.dropna(inplace=True)
	return data_df

def R_calculate(farg, *args):

    R = [np.corrcoef(r,y) for r in farg]

    return R  

def fillMed(farg):
	farg['BareNuc']=farg['BareNuc'].fillna(farg["BareNuc"].median())
	return farg


def main():
	dataset = 'WBCD.csv'
	data = read_data(dataset)
	data = clean_data(data)
	# exclude ID and class label
	d_matrix = data.values(columns=data.columns[1:-1])
	# extract class labels in y vector:
	y = data.values(columns=data.columns[-1:])



if __name__ == '__main__':
    main()
