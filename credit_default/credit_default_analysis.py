#!/usr/bin/env python
# coding: utf-8

# # Analyzing Credit Card Defaults for Taiwan clients

# ## 1. Experiment Objective

# I will be working on a dataset found on the UCI Machine Learning Repository. This dataset has information on default payments, demographic factors, credit data, history of payment, and bill statements of credit card clients in Taiwan from April 2005 to September 2005. I choose to analyze this dataset because I was interested in finding what signs there is to predict borrower from defaulting a month in advance. If that can be predicted accurately, lenders could decrease the avaliable limit or possibly close the account before it's used more.
# 
# The UCI Machine Learning Repository a web created by a graduate student and it is a collection of datasets that are used by the data science community for education purpose. There is no authentication required to access the datasets and are okay to be used for practice purposes.  
# 
# The provided definations of the features are as follow: 
# 
# - ID: ID of each client
# - LIMIT_BAL: Amount of given credit in NT dollars
# - SEX: Gender (1=male, 2=female)
# - EDUCATION: (1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown)
# - MARRIAGE: Marital status (1=married, 2=single, 3=others)
# - AGE: Age in years
# - PAY_0: Repayment status in September, 2005 (-1=pay duly, 1=payment delay for one month, 2=payment delay for two months, … 8=payment delay for eight months, 9=payment delay for nine months and above)
# - PAY_2: Repayment status in August, 2005 (scale same as above)
# - PAY_3: Repayment status in July, 2005 (scale same as above)
# - PAY_4: Repayment status in June, 2005 (scale same as above)
# - PAY_5: Repayment status in May, 2005 (scale same as above)
# - PAY_6: Repayment status in April, 2005 (scale same as above)
# - BILL_AMT1: Amount of bill statement in September, 2005 (NT dollar)
# - BILL_AMT2: Amount of bill statement in August, 2005 (NT dollar)
# - BILL_AMT3: Amount of bill statement in July, 2005 (NT dollar)
# - BILL_AMT4: Amount of bill statement in June, 2005 (NT dollar)
# - BILL_AMT5: Amount of bill statement in May, 2005 (NT dollar)
# - BILL_AMT6: Amount of bill statement in April, 2005 (NT dollar)
# - PAY_AMT1: Amount of previous payment in September, 2005 (NT dollar)
# - PAY_AMT2: Amount of previous payment in August, 2005 (NT dollar)
# - PAY_AMT3: Amount of previous payment in July, 2005 (NT dollar)
# - PAY_AMT4: Amount of previous payment in June, 2005 (NT dollar)
# - PAY_AMT5: Amount of previous payment in May, 2005 (NT dollar)
# - PAY_AMT6: Amount of previous payment in April, 2005 (NT dollar)
# - Default: Default payment (1=yes, 0=no)
# 
# 

# In[10]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
get_ipython().run_line_magic('matplotlib', 'inline')


# ## 2.  Data collection

# In[11]:


file = "default.xls"
df = pd.read_excel(file, header=1, index_col=0)
df.head()


# In[12]:


# check for data type and shape of the data 
df.info()


# The dataset contains 24 columns and 30,000 examples. All though the type of data for all columns are labeled integer, we can see we are working with some catagorical data that has been encoded to integer type. 
# 
# Before I move on to exploring and cleaning the dataset, I will start by cleaning the column names to make it easier to work with. 

# ## Clean Feature Names 

# In[16]:


print(df.columns)


# Changes to make based on initial observations:
# 
# - change the label name to make it shorter 
# - change the first feature name to describe the data more
# - convert all feature names to lower case 
# 
# 

# In[17]:


# clean column names to make it easy to work with 
def clean_cols(col):
    col = col.strip() # remove any white spaces 
    col = col.replace("default payment next month", "default")
    col = col.replace('LIMIT_BAL', 'approved_bal')
    col = col.lower()
    return col 


new_cols = []
for col in df.columns:
    clean_col = clean_cols(col)
    new_cols.append(clean_col)
    
df.columns = new_cols
print(df.columns)


# ## 3. Initial Data Exploration and Data Preprocessing
# 
# I will start by exploring the data to have a better understanding at what I am working with and if there is any features that is irrelevant to my problem. After removing the irrelevant features, I will start cleaning the dataset.  

# In[18]:


# check some statistical values for all columns
df.describe(include="all")


# In[19]:


# value counts for all features 
column_names = df.columns.tolist()


for column in column_names:
    print(column)
    print(df[column].value_counts(dropna=False))


# In[20]:


df.hist(bins=30 ,figsize =(20,15))
plt.show() 


# Initial observations:
# 
#     -  There are 6 columns that shows the status of the client for the prior 6 months
#     -  There are also another 6 columns showing the balance and payments made

# In[268]:


df['pay_0'].value_counts()


# In[21]:


sns.countplot(x = 'pay_0', data=df, palette='hls')
plt.show()


# In[22]:


# the average age is 35
df.age.mean()


# In[23]:


sns.countplot(x = 'age', data=df, palette='hls')
plt.show()


# In[24]:


# The average balance approved is 167,484
df.approved_bal.mean()


# In[25]:


sns.countplot(x = 'approved_bal', data=df, palette='hls')
plt.show()


# Since the aim of this project is to predict if a client will default the following month, I will only be using thier status and balance at the current month. I will drop the other 5 months. The model is attempting to predict next month defualt probability. We want to predict defualt probability based on the current status.

# In[26]:


df = df.drop(['pay_2', 'pay_3', 'pay_4', 'pay_5', 'pay_6', 
            'bill_amt2', 'bill_amt3', 'bill_amt4', 'bill_amt5', 'bill_amt6',
            'pay_amt2', 'pay_amt3', 'pay_amt4', 'pay_amt5', 'pay_amt6'], axis = 1)
df.head()


# We will rename the columns for the current status, current balance and payments made.

# In[27]:


df.rename(columns={'pay_0': 'status', 'bill_amt1': 'balance', 'pay_amt1': 'payment'},
          inplace=True)
df.head()


# In[28]:


# print value counts for the catagorical cols only
for col in df.columns[[1, 2, 3, 5, 6, 7]]:
    print(col)
    print(df[col].value_counts())


# ## Exploring status

# - After evaluating the value counts, we have some inconsistent data. For status we have -1, -2 labels which could mean they have no current balance. 
# - Since those values are closly related to 0 meaning they are current, we will replace -1 and -2 with 0. 
# - The fact that they have no balance is irrelevant for our analysis.

# In[29]:


# replace -1 and -2 with 0
df["status"] = df["status"].replace([-1, -2], 0)
df["status"].value_counts(dropna=False)


# ## Exploring education

# In[30]:


df["education"].value_counts(dropna=False)


# The labels given for education is as follow:
# 
# - 1 = graduate, 2 = undergrad, 3 = Highschool, 4 = others, 5, 6 and 0 are uknown or missing values.
# - The uknown values account for about 400 instances. 
# - Since we can't predict what those labels could be, we will remove those rows 

# In[31]:


# Remove rows containing 0, 4, 5, 6 for education 

df.drop(((df[df['education'] > 3].index) | (df[df['education'] == 0].index)), 
        inplace = True) 


# In[32]:


print(df.shape)
df["education"].value_counts(dropna=False)


# After cleaning the education feature, we now have lables 1, 2, 3 for graduate, undergrad and high school. 
# 
# Next lets work on the marriage feature. 
# 
# - Values given should be 1 = married, 2 = single, 3 = others 
# - We have about 54 rows with 0 label and we can convert those values to 3 

# In[33]:


df["marriage"].value_counts()


# In[34]:


df["marriage"] = df["marriage"].replace(0, 3)
df["marriage"].value_counts()


# ## Exploring balance, status and payment features

# In[35]:


(df[(df.balance <= 0) & (df.default == 1) & (df.status == 0)])


# After exploring more initial finding are:
# 
# - We have 181 columns with balance showing negative or 0, status is 0 (not behind on payments) and default 1. 
# - we will drop those rows since they are inconsistent

# In[36]:


df.drop(df[(df.balance <= 0) & (df.default == 1) & (df.status == 0)].index, inplace=True)


# We also have 360 rows that have instances making a payment, status is not zero suggesting they are behind but the balance is 0. 
# 
# We will drop those values as well. 

# In[37]:


df[(df.status != 0) & (df.balance == 0) & (df.payment != 0)]


# In[38]:


df.drop(df[(df.status != 0) & (df.balance == 0) & (df.payment != 0)].index, 
        inplace = True) 


# ## Remove Duplicates 

# In[39]:


duplicate_rows = df[df.duplicated()]
duplicate_rows.shape


# We have 124 rows of duplicated data. We don't need duplicates data so we will remove them.  

# In[40]:


# drop duplicates 
df = df.drop_duplicates()


# ## Feature engineering 
# 
# Adding new feature with the utilization rate could be good adition to the model.  

# In[41]:


df["utilization"] = (df["balance"] / df["approved_bal"])
df.sample(3)


# In[42]:


# reindex columns to move the label to the last column
column_names = ['approved_bal', 'sex', 'education', 'marriage', 'age', 'status',
       'balance', 'payment', 'utilization', 'default']
df = df.reindex(columns=column_names)
df.head(5)


# In[43]:


print(df.shape)
print(df.dtypes)
print(df.isnull().sum()) 
print(df.nunique())


# In[44]:


print(df.iloc[0])
print(df.describe(include='all'))
print(df.sample(5))


# After cleaning the data we are left with 28867 examples and 9 features. We can also see that we have no null values. We will save the cleaned dataframe and we will move on to feature encoding and scaling. 

# In[45]:


df.to_csv('credit_default_cleaned.csv',index=False)


# ## Visualizations

# In[46]:


fig, ax = plt.subplots()
df["age"].hist(color = '#0072BD', edgecolor = 'black', 
             grid = False)
ax.set_title("Borrower Age Distribution", fontsize = 12)
ax.set_xlabel("Age")
ax.set_ylabel("frequency")
plt.show()


# In[47]:


plt.scatter(df['age'], df['approved_bal'], color="blue",
            marker="1", s=30)
plt.xlabel('Age')
plt.ylabel('Approved amount')

plt.title('Approved amount by Age')
plt.legend()

plt.show()


# In[48]:


df.hist(bins=50 ,figsize =(20,15))
plt.show() 


# ## Encode Nominal and Ordinal features 
# 
# - we have the sex and marriage features as nominal 
# - we have education and status as ordinal features 
# 
# - Use OneHotEncoder to encode the nominal features
# - Use ordinal encoder to encode the ordinal features 
# 
# - Use Discretization to bucket the age groups with equal width uniformly 

# ### We will encode the Ordinal features first 

# In[49]:


encoder = OrdinalEncoder()

df.education = encoder.fit_transform(df.education.values.reshape(-1, 1))
df.status = encoder.fit_transform(df.status.values.reshape(-1, 1))


# In[50]:


print(df.education.unique())
print(df.status.unique())


# ### Bucket age groups into bins with equal width uniformly

# In[51]:


fig, ax = plt.subplots()
df["age"].hist(color = '#0072BD', edgecolor = 'black', 
             grid = False)
ax.set_title("Borrower Age Distribution", fontsize = 12)
ax.set_xlabel("Age")
ax.set_ylabel("frequency")
plt.show()


# In[52]:


discret = KBinsDiscretizer(n_bins=6, encode='ordinal', 
                        strategy='uniform')

df.age = discret.fit_transform(df.age.values.reshape(-1, 1))


# In[53]:


fig, ax = plt.subplots()
df["age"].hist(color = '#0072BD', edgecolor = 'black', 
             grid = False)
ax.set_title("Borrower Age Distribution", fontsize = 12)
ax.set_xlabel("Age")
ax.set_ylabel("frequency")
plt.show()


# ## Feature Scale 

# In[54]:


df.dtypes


# In[55]:


scaler = MinMaxScaler()
# transform data
scaled = scaler.fit_transform(df)
print(scaled[:5])


# ### Split data for training set and test set

# In[56]:


# split for train set and test set 
X = df.iloc[:, 0:-1]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, 
                                                    random_state=1, stratify = y)


# In[57]:


# stratify parameter is keeping the same proportion of labels as the dataset in test and train 
# 22% for default and 78% not default

print(np.bincount(y))
print(np.bincount(y_train))
print(np.bincount(y_test))


# In[58]:


print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)


# ### We will encode the Nominal features 
# 
# Since the values are already in integers, we just need to use OneHotEncoder to add the additional columns and we do not need to convert values to integers. 

# In[59]:


# use column tranformer to transform multiple columns at once 
onehot = OneHotEncoder(dtype=np.int, sparse=True)

# index 1 and 3 for the sex and marriage features 
col_transform = ColumnTransformer([('encoder', OneHotEncoder(), [1, 3])],
                                  remainder='passthrough')


# In[60]:


X_train = np.array(col_transform.fit_transform(X_train), dtype = np.int)
X_test = np.array(col_transform.transform(X_test), dtype = np.int)


# # 4. Model Comparison

# ## Logistic Regression Model

# In[61]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()


# In[62]:


lr.fit(X_train, y_train)


# In[63]:


predictions = lr.predict(X_test)


# In[64]:


# Use score method to get accuracy of model
score = lr.score(X_test, y_test)
print(score)


# In[65]:


from sklearn import metrics


# In[66]:


confusion = metrics.confusion_matrix(y_test, predictions)
print(confusion)


# In[67]:


plt.figure(figsize=(9,9))
sns.heatmap(confusion, annot=True, fmt=".3f", linewidths=.5, 
            square = True, cmap = 'Blues_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15)


# In[68]:


from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))


# ## Decision Tree 

# In[69]:


from sklearn import tree
model = tree.DecisionTreeClassifier(max_depth=3)


# In[70]:


model.fit(X_train, y_train)


# In[71]:


y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)


# In[72]:


tree.plot_tree(model)
plt.show()


# ## Random Forest

# In[73]:


from sklearn.ensemble import RandomForestClassifier

# Create the model with 30 trees
model = RandomForestClassifier(n_estimators=50, 
                               bootstrap = True,
                               max_features = 'sqrt')
# Fit on training data
model.fit(X_train, y_train)


# In[74]:


# Actual class predictions
y_pred = model.predict(X_test)


# Probabilities for each class
probs = model.predict_proba(X_test)[:, 1]


# In[75]:


accuracy_score(y_test, y_pred)


# In[76]:


confusion = metrics.confusion_matrix(y_test, y_pred)
print(confusion)


# In[77]:


plt.figure(figsize=(9,9))
sns.heatmap(confusion, annot=True, fmt=".3f", linewidths=.5, 
            square = True, cmap = 'Blues_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15)


# # 5. Experiment summary

# Data collection was done through the website of UCI Machine Learning Repository. The dataset was downloaded as excel file and read using the pandas excel reader. These dataset had lots of irrelevant features I was able to drop. All though this dataset did not have missing values, it did have many values that was inconsistent. After doing some analysis by comparing important features against each other using comparasion operators, I was able to filter out the inconsistent data. 
# 
# I used KBinsDiscretizer to put the age groups into bins and encoded the nominal and ordinal features. I used the min max scaler to scale some features as well before I started with my model comparison. 
# 
# I compared the performance of Logisitc Regression, Decision Tree and Random forest models. The Descion Tree model performed better based on accuracy using only depth of 3. Allthough they were all around 80% accuarte, I found that Decision tree performed better than the Random forest as well. I learned that hyperparameter tuning does help with accuracy. With the default depth, My accuracy was only 72% however after changing the depth to 3-5, the accuracy went up to 82% for all. 

# In[ ]:




