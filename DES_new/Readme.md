# DES_new Code introduction
Dynamic ensemble selection strategy for the Error-Correcting Output Codes Family.

This is the implementation for paper:

The Design of Dynamic Ensemble Selection Strategy for the Error-Correcting Output Codes Family.

## Environment
	Windows 10 64bit
	python 3.7.6
	scikit-learn 0.19.2
	numpy 1.15.4
	
## Data format
	Data should be put into the folder data/source.
	Each line is a sample.
	The last column represents the labels and the rest are feature space.
	There are fifteen examples data sets in the folder data/source.
	The invalid sample, such as value missed, will cause errors.

## Data storage
	data/source:	Storing source data.
	data/split:	Storing splited data.
	data/norm:	Storing normalized data.
	data/exp_mat:	Storing coding matrix.
	data/support:	Storing feature score.
	data/fea_num:	Storing feature selected number.
	data/exp_data:	Storing experimental data.
	data/exp:	Storing experimental results.

## Program entrance
	main.py:	the running code of the paper.

## Parameter setting
	main.py:
		exp_num:		Number of experiments.
		decoder_code:		Type of decoder.
		complexity_type:	Type of data complexity.
	Tools/FeatureSelect.py:
		alpha:			Iteration Step.
		beta:			Size of comparison interval.

## Run project
	Run the following command:	python main.py
	All result infos will be written into the folder 'data'
