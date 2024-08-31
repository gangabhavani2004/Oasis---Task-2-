# Task 2 - Project Title: Customer Segmentation

# Importing Libraries


```python
#Importing necessary libraries
import numpy as np # linear algebra
import pandas as pd # data processing
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pointbiserialr
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

```


```python
import warnings
warnings.filterwarnings("ignore")
```

# 1. Data Collection: Obtain a dataset containing customer information, purchase history, and relevant data.


```python
#Reading the data
data = pd.read_csv("C:/Users/GANGA/Desktop/Oyasis/ifood_df-Task2.csv")

#Taking a look at the top 5 rows of the data
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Income</th>
      <th>Kidhome</th>
      <th>Teenhome</th>
      <th>Recency</th>
      <th>MntWines</th>
      <th>MntFruits</th>
      <th>MntMeatProducts</th>
      <th>MntFishProducts</th>
      <th>MntSweetProducts</th>
      <th>MntGoldProds</th>
      <th>...</th>
      <th>marital_Together</th>
      <th>marital_Widow</th>
      <th>education_2n Cycle</th>
      <th>education_Basic</th>
      <th>education_Graduation</th>
      <th>education_Master</th>
      <th>education_PhD</th>
      <th>MntTotal</th>
      <th>MntRegularProds</th>
      <th>AcceptedCmpOverall</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>58138.0</td>
      <td>0</td>
      <td>0</td>
      <td>58</td>
      <td>635</td>
      <td>88</td>
      <td>546</td>
      <td>172</td>
      <td>88</td>
      <td>88</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1529</td>
      <td>1441</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>46344.0</td>
      <td>1</td>
      <td>1</td>
      <td>38</td>
      <td>11</td>
      <td>1</td>
      <td>6</td>
      <td>2</td>
      <td>1</td>
      <td>6</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>21</td>
      <td>15</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>71613.0</td>
      <td>0</td>
      <td>0</td>
      <td>26</td>
      <td>426</td>
      <td>49</td>
      <td>127</td>
      <td>111</td>
      <td>21</td>
      <td>42</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>734</td>
      <td>692</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>26646.0</td>
      <td>1</td>
      <td>0</td>
      <td>26</td>
      <td>11</td>
      <td>4</td>
      <td>20</td>
      <td>10</td>
      <td>3</td>
      <td>5</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>48</td>
      <td>43</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>58293.0</td>
      <td>1</td>
      <td>0</td>
      <td>94</td>
      <td>173</td>
      <td>43</td>
      <td>118</td>
      <td>46</td>
      <td>27</td>
      <td>15</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>407</td>
      <td>392</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 39 columns</p>
</div>



# 2. Data Exploration and Cleaning: Explore the dataset, understand its structure, and handle any missing or inconsistent data.


```python
data.columns
```




    Index(['Income', 'Kidhome', 'Teenhome', 'Recency', 'MntWines', 'MntFruits',
           'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts',
           'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases',
           'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth',
           'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1',
           'AcceptedCmp2', 'Complain', 'Z_CostContact', 'Z_Revenue', 'Response',
           'Age', 'Customer_Days', 'marital_Divorced', 'marital_Married',
           'marital_Single', 'marital_Together', 'marital_Widow',
           'education_2n Cycle', 'education_Basic', 'education_Graduation',
           'education_Master', 'education_PhD', 'MntTotal', 'MntRegularProds',
           'AcceptedCmpOverall'],
          dtype='object')




```python
# Check data types of each column
data.dtypes
```




    Income                  float64
    Kidhome                   int64
    Teenhome                  int64
    Recency                   int64
    MntWines                  int64
    MntFruits                 int64
    MntMeatProducts           int64
    MntFishProducts           int64
    MntSweetProducts          int64
    MntGoldProds              int64
    NumDealsPurchases         int64
    NumWebPurchases           int64
    NumCatalogPurchases       int64
    NumStorePurchases         int64
    NumWebVisitsMonth         int64
    AcceptedCmp3              int64
    AcceptedCmp4              int64
    AcceptedCmp5              int64
    AcceptedCmp1              int64
    AcceptedCmp2              int64
    Complain                  int64
    Z_CostContact             int64
    Z_Revenue                 int64
    Response                  int64
    Age                       int64
    Customer_Days             int64
    marital_Divorced          int64
    marital_Married           int64
    marital_Single            int64
    marital_Together          int64
    marital_Widow             int64
    education_2n Cycle        int64
    education_Basic           int64
    education_Graduation      int64
    education_Master          int64
    education_PhD             int64
    MntTotal                  int64
    MntRegularProds           int64
    AcceptedCmpOverall        int64
    dtype: object




```python
# Summary statistics for numerical columns
data.describe()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Income</th>
      <th>Kidhome</th>
      <th>Teenhome</th>
      <th>Recency</th>
      <th>MntWines</th>
      <th>MntFruits</th>
      <th>MntMeatProducts</th>
      <th>MntFishProducts</th>
      <th>MntSweetProducts</th>
      <th>MntGoldProds</th>
      <th>...</th>
      <th>marital_Together</th>
      <th>marital_Widow</th>
      <th>education_2n Cycle</th>
      <th>education_Basic</th>
      <th>education_Graduation</th>
      <th>education_Master</th>
      <th>education_PhD</th>
      <th>MntTotal</th>
      <th>MntRegularProds</th>
      <th>AcceptedCmpOverall</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2205.000000</td>
      <td>2205.000000</td>
      <td>2205.000000</td>
      <td>2205.000000</td>
      <td>2205.000000</td>
      <td>2205.000000</td>
      <td>2205.000000</td>
      <td>2205.000000</td>
      <td>2205.000000</td>
      <td>2205.000000</td>
      <td>...</td>
      <td>2205.000000</td>
      <td>2205.000000</td>
      <td>2205.000000</td>
      <td>2205.000000</td>
      <td>2205.000000</td>
      <td>2205.000000</td>
      <td>2205.000000</td>
      <td>2205.000000</td>
      <td>2205.000000</td>
      <td>2205.00000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>51622.094785</td>
      <td>0.442177</td>
      <td>0.506576</td>
      <td>49.009070</td>
      <td>306.164626</td>
      <td>26.403175</td>
      <td>165.312018</td>
      <td>37.756463</td>
      <td>27.128345</td>
      <td>44.057143</td>
      <td>...</td>
      <td>0.257596</td>
      <td>0.034467</td>
      <td>0.089796</td>
      <td>0.024490</td>
      <td>0.504762</td>
      <td>0.165079</td>
      <td>0.215873</td>
      <td>562.764626</td>
      <td>518.707483</td>
      <td>0.29932</td>
    </tr>
    <tr>
      <th>std</th>
      <td>20713.063826</td>
      <td>0.537132</td>
      <td>0.544380</td>
      <td>28.932111</td>
      <td>337.493839</td>
      <td>39.784484</td>
      <td>217.784507</td>
      <td>54.824635</td>
      <td>41.130468</td>
      <td>51.736211</td>
      <td>...</td>
      <td>0.437410</td>
      <td>0.182467</td>
      <td>0.285954</td>
      <td>0.154599</td>
      <td>0.500091</td>
      <td>0.371336</td>
      <td>0.411520</td>
      <td>575.936911</td>
      <td>553.847248</td>
      <td>0.68044</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1730.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>4.000000</td>
      <td>-283.000000</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>35196.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>24.000000</td>
      <td>24.000000</td>
      <td>2.000000</td>
      <td>16.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>9.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>56.000000</td>
      <td>42.000000</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>51287.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>49.000000</td>
      <td>178.000000</td>
      <td>8.000000</td>
      <td>68.000000</td>
      <td>12.000000</td>
      <td>8.000000</td>
      <td>25.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>343.000000</td>
      <td>288.000000</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>68281.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>74.000000</td>
      <td>507.000000</td>
      <td>33.000000</td>
      <td>232.000000</td>
      <td>50.000000</td>
      <td>34.000000</td>
      <td>56.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>964.000000</td>
      <td>884.000000</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>113734.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>99.000000</td>
      <td>1493.000000</td>
      <td>199.000000</td>
      <td>1725.000000</td>
      <td>259.000000</td>
      <td>262.000000</td>
      <td>321.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>2491.000000</td>
      <td>2458.000000</td>
      <td>4.00000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 39 columns</p>
</div>




```python
data.isnull().sum()
```




    Income                  0
    Kidhome                 0
    Teenhome                0
    Recency                 0
    MntWines                0
    MntFruits               0
    MntMeatProducts         0
    MntFishProducts         0
    MntSweetProducts        0
    MntGoldProds            0
    NumDealsPurchases       0
    NumWebPurchases         0
    NumCatalogPurchases     0
    NumStorePurchases       0
    NumWebVisitsMonth       0
    AcceptedCmp3            0
    AcceptedCmp4            0
    AcceptedCmp5            0
    AcceptedCmp1            0
    AcceptedCmp2            0
    Complain                0
    Z_CostContact           0
    Z_Revenue               0
    Response                0
    Age                     0
    Customer_Days           0
    marital_Divorced        0
    marital_Married         0
    marital_Single          0
    marital_Together        0
    marital_Widow           0
    education_2n Cycle      0
    education_Basic         0
    education_Graduation    0
    education_Master        0
    education_PhD           0
    MntTotal                0
    MntRegularProds         0
    AcceptedCmpOverall      0
    dtype: int64




```python
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2205 entries, 0 to 2204
    Data columns (total 39 columns):
     #   Column                Non-Null Count  Dtype  
    ---  ------                --------------  -----  
     0   Income                2205 non-null   float64
     1   Kidhome               2205 non-null   int64  
     2   Teenhome              2205 non-null   int64  
     3   Recency               2205 non-null   int64  
     4   MntWines              2205 non-null   int64  
     5   MntFruits             2205 non-null   int64  
     6   MntMeatProducts       2205 non-null   int64  
     7   MntFishProducts       2205 non-null   int64  
     8   MntSweetProducts      2205 non-null   int64  
     9   MntGoldProds          2205 non-null   int64  
     10  NumDealsPurchases     2205 non-null   int64  
     11  NumWebPurchases       2205 non-null   int64  
     12  NumCatalogPurchases   2205 non-null   int64  
     13  NumStorePurchases     2205 non-null   int64  
     14  NumWebVisitsMonth     2205 non-null   int64  
     15  AcceptedCmp3          2205 non-null   int64  
     16  AcceptedCmp4          2205 non-null   int64  
     17  AcceptedCmp5          2205 non-null   int64  
     18  AcceptedCmp1          2205 non-null   int64  
     19  AcceptedCmp2          2205 non-null   int64  
     20  Complain              2205 non-null   int64  
     21  Z_CostContact         2205 non-null   int64  
     22  Z_Revenue             2205 non-null   int64  
     23  Response              2205 non-null   int64  
     24  Age                   2205 non-null   int64  
     25  Customer_Days         2205 non-null   int64  
     26  marital_Divorced      2205 non-null   int64  
     27  marital_Married       2205 non-null   int64  
     28  marital_Single        2205 non-null   int64  
     29  marital_Together      2205 non-null   int64  
     30  marital_Widow         2205 non-null   int64  
     31  education_2n Cycle    2205 non-null   int64  
     32  education_Basic       2205 non-null   int64  
     33  education_Graduation  2205 non-null   int64  
     34  education_Master      2205 non-null   int64  
     35  education_PhD         2205 non-null   int64  
     36  MntTotal              2205 non-null   int64  
     37  MntRegularProds       2205 non-null   int64  
     38  AcceptedCmpOverall    2205 non-null   int64  
    dtypes: float64(1), int64(38)
    memory usage: 672.0 KB
    


```python
data.nunique()
```




    Income                  1963
    Kidhome                    3
    Teenhome                   3
    Recency                  100
    MntWines                 775
    MntFruits                158
    MntMeatProducts          551
    MntFishProducts          182
    MntSweetProducts         176
    MntGoldProds             212
    NumDealsPurchases         15
    NumWebPurchases           15
    NumCatalogPurchases       13
    NumStorePurchases         14
    NumWebVisitsMonth         16
    AcceptedCmp3               2
    AcceptedCmp4               2
    AcceptedCmp5               2
    AcceptedCmp1               2
    AcceptedCmp2               2
    Complain                   2
    Z_CostContact              1
    Z_Revenue                  1
    Response                   2
    Age                       56
    Customer_Days            662
    marital_Divorced           2
    marital_Married            2
    marital_Single             2
    marital_Together           2
    marital_Widow              2
    education_2n Cycle         2
    education_Basic            2
    education_Graduation       2
    education_Master           2
    education_PhD              2
    MntTotal                 897
    MntRegularProds          974
    AcceptedCmpOverall         5
    dtype: int64




```python
# Handle missing values (drop rows with missing values)
data = data.dropna()

```


```python
# Remove duplicates
data = data.drop_duplicates()

```


```python
data.drop(columns=['Z_CostContact','Z_Revenue'],inplace=True)
data.columns
```




    Index(['Income', 'Kidhome', 'Teenhome', 'Recency', 'MntWines', 'MntFruits',
           'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts',
           'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases',
           'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth',
           'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1',
           'AcceptedCmp2', 'Complain', 'Response', 'Age', 'Customer_Days',
           'marital_Divorced', 'marital_Married', 'marital_Single',
           'marital_Together', 'marital_Widow', 'education_2n Cycle',
           'education_Basic', 'education_Graduation', 'education_Master',
           'education_PhD', 'MntTotal', 'MntRegularProds', 'AcceptedCmpOverall'],
          dtype='object')



# 3. Descriptive Statistics: Calculate key metrics such as average purchase value, frequency of purchases, etc.



1. Calculate the Average Purchase Value


```python
# Calculate the total amount spent by each customer (if not already available)
data['Total_Spent'] = data['MntWines'] + data['MntFruits'] + data['MntMeatProducts'] + data['MntFishProducts'] + data['MntSweetProducts'] + data['MntGoldProds']

# Calculate the average purchase value
data['Average_Purchase_Value'] = data['Total_Spent'] / (data['NumDealsPurchases'] + data['NumWebPurchases'] + data['NumCatalogPurchases'] + data['NumStorePurchases'])

# Display the first few rows to verify
print(data[['Total_Spent', 'Average_Purchase_Value']].head())

```

       Total_Spent  Average_Purchase_Value
    0         1617               64.680000
    1           27                4.500000
    2          776               36.952381
    3           53                6.625000
    4          422               22.210526
    

2. Calculate the Frequency of Purchases


```python
# Calculate the total number of purchases (Frequency)
data['Total_Purchases'] = data['NumDealsPurchases'] + data['NumWebPurchases'] + data['NumCatalogPurchases'] + data['NumStorePurchases']

# Display the first few rows to verify
print(data[['Total_Purchases']].head())

```

       Total_Purchases
    0               25
    1                6
    2               21
    3                8
    4               19
    

3. Calculate Recency


```python
# Display the Recency column
print(data['Recency'].describe())

```

    count    2021.000000
    mean       48.880752
    std        28.950917
    min         0.000000
    25%        24.000000
    50%        49.000000
    75%        74.000000
    max        99.000000
    Name: Recency, dtype: float64
    

4. Descriptive Statistics for Key Metrics


```python
# Descriptive statistics for spending metrics
spending_columns = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds', 'Total_Spent', 'Average_Purchase_Value']

# Calculate descriptive statistics
spending_stats = data[spending_columns].describe()

# Display the statistics
print(spending_stats)

```

              MntWines    MntFruits  MntMeatProducts  MntFishProducts  \
    count  2021.000000  2021.000000      2021.000000      2021.000000   
    mean    306.492331    26.364671       166.059871        37.603662   
    std     337.603877    39.776518       219.869126        54.892196   
    min       0.000000     0.000000         0.000000         0.000000   
    25%      24.000000     2.000000        16.000000         3.000000   
    50%     178.000000     8.000000        68.000000        12.000000   
    75%     507.000000    33.000000       230.000000        50.000000   
    max    1493.000000   199.000000      1725.000000       259.000000   
    
           MntSweetProducts  MntGoldProds  Total_Spent  Average_Purchase_Value  
    count       2021.000000   2021.000000  2021.000000             2021.000000  
    mean          27.268679     43.921821   607.711034                     inf  
    std           41.575454     51.678211   602.396167                     NaN  
    min            0.000000      0.000000     5.000000                0.533333  
    25%            1.000000      9.000000    69.000000                9.714286  
    50%            8.000000     25.000000   397.000000               23.681818  
    75%           34.000000     56.000000  1048.000000               45.640000  
    max          262.000000    321.000000  2525.000000                     inf  
    

5. Customer Segmentation Metrics


```python
# Example: Count of customers in different income brackets
income_brackets = pd.cut(data['Income'], bins=[0, 30000, 60000, 90000, 120000], labels=['Low', 'Medium', 'High', 'Very High'])
income_distribution = income_brackets.value_counts()

# Display income distribution
print(income_distribution)

# Example: Count of customers who responded to campaigns
campaign_response = data[['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5']].sum(axis=1)
print(campaign_response.value_counts())

```

    Income
    Medium       925
    High         716
    Low          337
    Very High     43
    Name: count, dtype: int64
    0    1595
    1     301
    2      75
    3      40
    4      10
    Name: count, dtype: int64
    

6. Visualize the Descriptive Statistics


```python
# Distribution of Total Spending
sns.histplot(data['Total_Spent'], kde=True)
plt.title('Distribution of Total Spending')
plt.xlabel('Total Spending')
plt.ylabel('Frequency')
plt.show()
```


    
![png](output_28_0.png)
    



```python
#Boxplot of Average Purchase Value
sns.boxplot(x='Average_Purchase_Value', data=data)
plt.title('Boxplot of Average Purchase Value')
plt.show()
```


    
![png](output_29_0.png)
    


# 4.Customer Segmentation: Utilize clustering algorithms (e.g., K-means) to segment customers based on behavior and purchase patterns.

#Select Features for Clustering


```python
# Select the relevant features for clustering
features = ['Income', 'Kidhome', 'Teenhome', 'Recency', 'MntWines', 'MntFruits',
            'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds',
            'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases',
            'NumWebVisitsMonth', 'Age', 'Total_Spent', 'Average_Purchase_Value', 'Total_Purchases']

# Create a new DataFrame with the selected features
df_clustering = data[features].copy()

```

Standardize the Data


```python
# Check the data types of the columns in df_clustering
print(df_clustering.dtypes)

# Convert any non-numeric columns to numeric if necessary
# Example: Converting categorical columns using one-hot encoding (if needed)
# df_clustering = pd.get_dummies(df_clustering)



```

    Income                    float64
    Kidhome                     int64
    Teenhome                    int64
    Recency                     int64
    MntWines                    int64
    MntFruits                   int64
    MntMeatProducts             int64
    MntFishProducts             int64
    MntSweetProducts            int64
    MntGoldProds                int64
    NumDealsPurchases           int64
    NumWebPurchases             int64
    NumCatalogPurchases         int64
    NumStorePurchases           int64
    NumWebVisitsMonth           int64
    Age                         int64
    Total_Spent                 int64
    Average_Purchase_Value    float64
    Total_Purchases             int64
    dtype: object
    


```python

# Check for missing values in the dataset
print(df_clustering.isnull().sum())

# Option 1: Fill missing values
df_clustering.fillna(df_clustering.median(), inplace=True)

# Option 2: Drop rows with missing values
# df_clustering.dropna(inplace=True)


```

    Income                    0
    Kidhome                   0
    Teenhome                  0
    Recency                   0
    MntWines                  0
    MntFruits                 0
    MntMeatProducts           0
    MntFishProducts           0
    MntSweetProducts          0
    MntGoldProds              0
    NumDealsPurchases         0
    NumWebPurchases           0
    NumCatalogPurchases       0
    NumStorePurchases         0
    NumWebVisitsMonth         0
    Age                       0
    Total_Spent               0
    Average_Purchase_Value    0
    Total_Purchases           0
    dtype: int64
    


```python
# Check for infinite values
print(np.isinf(df_clustering).sum())

# Replace infinite values with NaN (if any)
df_clustering.replace([np.inf, -np.inf], np.nan, inplace=True)

# After replacing, fill or drop NaNs as done before
df_clustering.fillna(df_clustering.median(), inplace=True)

```

    Income                    0
    Kidhome                   0
    Teenhome                  0
    Recency                   0
    MntWines                  0
    MntFruits                 0
    MntMeatProducts           0
    MntFishProducts           0
    MntSweetProducts          0
    MntGoldProds              0
    NumDealsPurchases         0
    NumWebPurchases           0
    NumCatalogPurchases       0
    NumStorePurchases         0
    NumWebVisitsMonth         0
    Age                       0
    Total_Spent               0
    Average_Purchase_Value    2
    Total_Purchases           0
    dtype: int64
    

Determine the Optimal Number of Clusters


```python
# Standardize the features after handling non-numeric, missing, and infinite values
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_clustering)

# Convert back to DataFrame for easier handling
df_scaled = pd.DataFrame(df_scaled, columns=features)

```


```python
# Elbow method to find the optimal number of clusters
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_scaled)
    sse.append(kmeans.inertia_)

# Plotting the Elbow Curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), sse, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters')
plt.ylabel('SSE (Sum of Squared Errors)')
plt.show()

```


    
![png](output_39_0.png)
    


Apply K-means Clustering


```python
# Apply K-means with the optimal number of clusters (e.g., 4)
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
data['Cluster'] = kmeans.fit_predict(df_scaled)

# View the number of customers in each segment
print(data['Cluster'].value_counts())

```

    Cluster
    0    824
    2    417
    1    407
    3    373
    Name: count, dtype: int64
    

Analyze and Visualize the Clusters


```python
# Add the cluster labels to the original DataFrame
data['Cluster'] = kmeans.labels_

# Calculate the average values for each cluster
cluster_analysis = data.groupby('Cluster').mean()

# Display the cluster analysis
print(cluster_analysis)

# Visualize the distribution of customers across clusters
plt.figure(figsize=(10, 6))
sns.countplot(data['Cluster'])
plt.title('Distribution of Customers Across Clusters')
plt.xlabel('Cluster')
plt.ylabel('Number of Customers')
plt.show()

```

                   Income   Kidhome  Teenhome    Recency    MntWines  MntFruits  \
    Cluster                                                                       
    0        32842.990291  0.824029  0.406553  48.747573   29.775485   4.400485   
    1        51323.248157  0.405405  0.882064  47.773956  259.167076  12.683047   
    2        77727.808153  0.023981  0.081535  49.059952  642.784173  69.153477   
    3        64601.254692  0.112601  0.809651  50.182306  593.469169  41.978552   
    
             MntMeatProducts  MntFishProducts  MntSweetProducts  MntGoldProds  \
    Cluster                                                                     
    0              19.078883         6.266990          4.582524     13.956311   
    1              79.486486        18.304668         12.297297     42.469287   
    2             507.942446       102.896882         71.652278     75.220624   
    3             203.010724        54.892761         44.101877     76.713137   
    
             ...  education_Basic  education_Graduation  education_Master  \
    Cluster  ...                                                            
    0        ...         0.057039              0.486650          0.158981   
    1        ...         0.002457              0.476658          0.206388   
    2        ...         0.000000              0.549161          0.163070   
    3        ...         0.002681              0.512064          0.139410   
    
             education_PhD     MntTotal  MntRegularProds  AcceptedCmpOverall  \
    Cluster                                                                    
    0             0.184466    64.104369        50.148058            0.088592   
    1             0.248157   381.938575       339.469287            0.194103   
    2             0.211031  1394.429257      1319.208633            0.760192   
    3             0.262735   937.453083       860.739946            0.380697   
    
             Total_Spent  Average_Purchase_Value  Total_Purchases  
    Cluster                                                        
    0          78.060680                     inf         7.245146  
    1         424.407862               24.163550        17.211302  
    2        1469.649880               75.647725        20.028777  
    3        1014.166220               43.982975        23.541555  
    
    [4 rows x 40 columns]
    


    
![png](output_43_1.png)
    


Interpret the Results


```python
# Print the cluster centers (mean values of features for each cluster)
print(kmeans.cluster_centers_)

# You can also plot features for each cluster to visualize differences
sns.boxplot(x='Cluster', y='Total_Spent', data=data)
plt.title('Total Spent by Cluster')
plt.show()

```

    [[-0.9100029   0.71014756 -0.18873005 -0.00460131 -0.81985235 -0.55232641
      -0.66865852 -0.5710179  -0.54579727 -0.57999157 -0.25136783 -0.80100622
      -0.78244961 -0.86140867  0.47553188 -0.31436871 -0.87945686 -0.78888787
      -1.00737843]
     [-0.01757833 -0.0707742   0.68175702 -0.03823956 -0.14021452 -0.34404747
      -0.39384707 -0.35166692 -0.36019058 -0.02811423  0.9045743   0.41597954
      -0.26024145  0.05731544  0.33876009  0.3273843  -0.30436538 -0.29740351
       0.30458279]
     [ 1.25751634 -0.78230283 -0.78372116  0.00619132  0.99636022  1.07599655
       1.55532154  1.18977532  1.06780756  0.60579785 -0.66231033  0.24558677
       1.23925741  0.69371326 -1.15465052  0.02078433  1.43120461  1.49719586
       0.67547979]
     [ 0.62362589 -0.61698714  0.54919619  0.04496837  0.85025049  0.39263732
       0.16809999  0.31504257  0.40498329  0.6346859   0.30870982  1.04106104
       0.62704133  1.06486578 -0.12928783  0.31401433  0.67489769  0.39344815
       1.13790765]]
    


    
![png](output_45_1.png)
    



```python

```


```python

```


```python

```


```python

```
