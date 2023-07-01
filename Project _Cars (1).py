#!/usr/bin/env python
# coding: utf-8

# In[588]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[589]:


df = pd.read_csv('cars.csv')


# In[590]:


df.head(11)


# In[591]:


df.sample(5)


# In[592]:


df.shape


# In[593]:


df.info()


# In[594]:


df.describe() #for int,float only


# In[595]:


df.describe(include='object')


# In[596]:


df.dropna(inplace=True)


# In[597]:


df.info()


# In[598]:


df.describe()


# In[599]:


df.describe(include='object')


# In[646]:


df.corr(method ='pearson')


# In[601]:


df[['km_driven']].describe().transpose()


# In[602]:


df[['km_driven']].sample(5)


# In[603]:


df[['km_driven']].max()


# In[604]:


sns.histplot(df['km_driven'], kde=True, legend=True)
plt.show()


# In[605]:


df.sort_values(by=['km_driven'],ascending=True).plot.scatter('km_driven','selling_price');


# In[606]:


df[['transmission', 'mileage']].describe().transpose()


# In[607]:


sns.histplot(df['transmission'])
plt.show()


# In[608]:


df['transmission'].value_counts()


# In[609]:


df.sort_values(by=['transmission'],ascending=True).plot.scatter('transmission','selling_price');


# In[610]:


df[['mileage']].sample(5)


# In[611]:



def return_mileage(mileage):
    return mileage.split(' ')[0]
df['mileage'] = df.mileage.apply(return_mileage)
df.sample(5)
#sns.histplot(df['split_mileage'])
#plt.show()


# In[612]:


df['mileage'] = df['mileage'].astype(float)


# In[613]:


df['fuel'].value_counts()


# In[614]:


df = df.drop(df[df.fuel == 'CNG'].index)


# In[615]:


df = df.drop(df[df.fuel == 'LPG'].index)


# In[616]:


df.describe() #изменения незначительные


# In[617]:


df[['mileage']].describe().transpose()


# In[618]:


df['mileage'].quantile(0.9)


# In[619]:


df['mileage'].plot.box();


# In[620]:


df.sort_values(by=['mileage'],ascending=True).plot.scatter('mileage','selling_price');


# In[621]:


df['name'].describe().transpose()


# In[622]:


sns.histplot(df['name'])
plt.show()


# In[623]:



def return_model(name):
    return name.split(' ')[0]
df['name'] = df.name.apply(return_model)
df.sample(5)


# In[624]:


df.sort_values(by=['name'], ascending=True).head()


# In[625]:


sns.histplot(df['name'].sort_values(ascending=True))
plt.show()


# In[626]:


df.sort_values(by=['name'],ascending=True).plot.scatter('name','selling_price');


# In[628]:


df['year'].describe()


# In[629]:


sns.histplot(df['year'])
plt.show()


# In[630]:


df.sort_values(by=['year'],ascending=True).plot.scatter('year','selling_price');


# In[632]:


df['owner'].describe()


# In[633]:


df['owner'].unique()


# In[634]:


df['owner'].value_counts()


# In[635]:


df = df.drop(df[df.owner == 'Test Drive Car'].index)


# In[636]:


df['owner'].value_counts() # проверка


# In[637]:


sns.histplot(df['owner'])
plt.show()


# In[638]:


df.plot.scatter('owner','selling_price');


# In[639]:


X = df.drop(['year','owner','name','selling_price', 'fuel', 'seller_type', 'engine', 'max_power','torque', 'seats'], axis=1)

y = df['selling_price'] # целевая переменная (target)


# In[640]:


X.head()


# In[641]:


X['transmission'] = X['transmission'].map({'Automatic' : 1, 'Manual' : 0})


# In[642]:


X.head()


# In[643]:


df.to_csv('carsnew.csv')


# In[644]:


X.dtypes


# In[645]:


X.info()


# In[648]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
np.array(X_train).reshape((-1, 1)), np.array(X_test).shape


# In[649]:


from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(X_train, y_train)


# In[650]:


r_sq = model.score(X_train, y_train)
print(f"coefficient of determination: {r_sq}")


# In[651]:


print(f"intercept: {model.intercept_}") #b0


# In[652]:


print(f"coefficients: {model.coef_}")  # b1


# In[653]:


pred = model.predict(X_test)


# In[654]:


print(f"predicted response:\n{pred[:11]}") #first 10


# In[657]:


import pickle

with open('model.pickle', 'wb') as f:
    pickle.dump(model, f)

