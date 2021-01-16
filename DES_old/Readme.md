# DES_old Code introduction
Dynamic ensemble selection strategy to improve Error Correcting Output Codes Algorithm.

This is the implementation for paper:

"J. Y. Zou, K. H. Liu, and Y. F. Huang, "A Dynamic Ensemble Selection Strategy for Improving Error Correcting Output Codes Algorithm," in 2019 IEEE Intl Conf on Parallel & Distributed Processing with Applications, Big Data & Cloud Computing, Sustainable Computing & Communications, Social Computing & Networking (ISPA/BDCloud/SocialCom/SustainCom), 2020."

## Environment
	Windows 10 64bit
	python 3.7.6
	scikit-learn 0.19.2
	numpy 1.15.4
	
## Data format
	Data should be put into the folder data/source.
	Each line is a sample.
	The last column represents the labels and the rest are feature space.
	There are seven examples data sets in the folder data/source.
	The invalid sample, such as value missed, will cause errors.

## Data storage
	data/source:	Storing source data.
	data/split:	Storing splited data.
	data/norm:    	Storing normalized data.
	data/exp_mat:	Storing coding matrix.
	data/support:	Storing feature score.
	data/fea_num: 	Storing feature selected number.
	data/exp_data:  Storing experimental data.
	data/exp:	Storing experimental results.

## Program entrance
	main.py:	the running code of the conference paper

## Run project
	Run the following command:	python main.py
	All result infos will be written into the folder 'data'
