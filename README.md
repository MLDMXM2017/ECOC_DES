# ECOC_DES

## Difference between DES_old and DES_new

### The implementation for paper

#### DES_old:
"J. Y. Zou, K. H. Liu, and Y. F. Huang, "A Dynamic Ensemble Selection Strategy for Improving Error Correcting Output Codes Algorithm," in 2019 IEEE Intl Conf on Parallel & Distributed Processing with Applications, Big Data & Cloud Computing, Sustainable Computing & Communications, Social Computing & Networking (ISPA/BDCloud/SocialCom/SustainCom), 2020."

#### DES_new
"The Design of Dynamic Ensemble Selection Strategy for the Error-Correcting Output Codes Family".

### DES_new's new contribution to DES_old:
    
#### Algorithm
    Choose feature subsets with data complexity instead of classifier accuracy  —— As shown in line 70, main.py and function "get_complexity" in DataComplexity/Get_Complexity.py
    Improve Fisher measure with Gaussian probability density function           —— As shown in line 95, main.py and function "fisher_gaussia_measure" in Tools/Distance.py
    Add weights to the column classifier during decoding                        —— As shown in line 120, main.py

#### Experiment
    More diverse data set
    More classic ECOC algorithms
    More types of base classifier
    More Analytical perspectives
