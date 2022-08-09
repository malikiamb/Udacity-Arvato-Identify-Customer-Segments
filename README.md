# Udacity-Arvato-Identify-Customer-Segments
Explore real customer demographics data from Bertelsmann partners AZ Direct and Arvato Finance Solution. This company performs mail-orders sales in Germany. We will utilize machine learning to segment data and make recommendations.
## Libraries used 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.ticker import PercentFormatter
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy import stats

# inline visualizations in notebook
%matplotlib inline


## Datasets 

Udacity_AZDIAS_Subset.csv: Demographics data for the general population of Germany; 891211 persons (rows) x 85 features (columns).  

Udacity_CUSTOMERS_Subset.csv: Demographics data for customers of a mail-order company; 191652 persons (rows) x 85 features (columns).

Data_Dictionary.md: Detailed information file about the features in the 
provided datasets.

AZDIAS_Feature_Summary.csv: Summary of feature attributes for demographics data; 85 features (rows) x 4 columns.

Each row of the demographics files represents a single person, but also 
includes information outside of individuals, including information about their household, building, and neighborhood.

Wrangling steps involved adjusting data types, reviewing outliers, null 
values, dropping columns that were not going to be utilized and adding features for ease of analysis.
Visualizations like bar charts histograms and countplots help to further understand the data.

# Summary of Findings

Initial programmatic assessment revealed that customer demographics dataset included many missing values, some features were binary but not numeric and additional features needed to be encoded to further understand data.
There are 49 Ordinal, 18 categorical, 6 mixed, and 6 numerical values

Feature transformation reduced dimensionality and identified relationships between variables in the dataset

Scaling the data helped to standardize the datasets 
PCA reduced dimensionality of datasets and allowed evaluation of variability of dataset
    90 components explained 84% of variability
k-means played integral role in clustering data to find similarities 
    elbow method produced k=12 clusters 

# Final Recommendation
We recommend cluster 9 as a target customer base for Arvarto mail order advertising since this cluster overrepresented customers vs. population

Further analysis of this cluster revealed like features like retired or near retirement, aspiring or likely to stay home, low interest in social activities as an example. Each of these would support the notion that this group has high potential to be available and respond to the marketing efforts. 


