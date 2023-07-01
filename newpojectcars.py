
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from PIL import Image


df = pd.read_csv('cars.csv')


df.head(11)



df.sample(5)



df.shape


df.info()



df.describe() #for int,float only

df.describe(include='object')


df.dropna(inplace=True)


df.info()



df.describe()




df.describe(include='object')

df.corr(method ='pearson')


df[['km_driven']].describe().transpose()


df[['km_driven']].sample(5)

df[['km_driven']].max()


sns.histplot(df['km_driven'], kde=True, legend=True)
plt.show()

df.sort_values(by=['km_driven'],ascending=True).plot.scatter('km_driven','selling_price');


df[['transmission', 'mileage']].describe().transpose()


sns.histplot(df['transmission'])
plt.show()

df['transmission'].value_counts()


df.sort_values(by=['transmission'],ascending=True).plot.scatter('transmission','selling_price');


df[['mileage']].sample(5)




def return_mileage(mileage):
    return mileage.split(' ')[0]
df['mileage'] = df.mileage.apply(return_mileage)
df.sample(5)
#sns.histplot(df['split_mileage'])
#plt.show()

df['mileage'] = df['mileage'].astype(float)
df['fuel'].value_counts()

df = df.drop(df[df.fuel == 'CNG'].index)


df = df.drop(df[df.fuel == 'LPG'].index)

df.describe() #изменения незначительные


df[['mileage']].describe().transpose()

df['mileage'].quantile(0.9)


df['mileage'].plot.box();


df.sort_values(by=['mileage'],ascending=True).plot.scatter('mileage','selling_price');


df['name'].describe().transpose()



sns.histplot(df['name'])
plt.show()


def return_model(name):
    return name.split(' ')[0]
df['name'] = df.name.apply(return_model)
df.sample(5)

df.sort_values(by=['name'], ascending=True).head()


sns.histplot(df['name'].sort_values(ascending=True))
plt.show()
df.sort_values(by=['name'],ascending=True).plot.scatter('name','selling_price');

df['year'].describe()


sns.histplot(df['year'])
plt.show()



df.sort_values(by=['year'],ascending=True).plot.scatter('year','selling_price');

df['owner'].describe()


df['owner'].unique()


df['owner'].value_counts()



df = df.drop(df[df.owner == 'Test Drive Car'].index)


df['owner'].value_counts() # проверка


sns.histplot(df['owner'])
plt.show()


df.plot.scatter('owner','selling_price');

