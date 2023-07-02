
import numpy as np
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




df[['km_driven']].describe().transpose()


df[['km_driven']].sample(5)

df[['km_driven']].max()






df[['transmission', 'mileage']].describe().transpose()




df['transmission'].value_counts()



df[['mileage']].sample(5)




def return_mileage(mileage):
    return mileage.split(' ')[0]
df['mileage'] = df.mileage.apply(return_mileage)
df.sample(5)


df['mileage'] = df['mileage'].astype(float)
df['fuel'].value_counts()

df = df.drop(df[df.fuel == 'CNG'].index)


df = df.drop(df[df.fuel == 'LPG'].index)

df.describe() #изменения незначительные


df[['mileage']].describe().transpose()

df['mileage'].quantile(0.9)






df['name'].describe().transpose()




def return_model(name):
    return name.split(' ')[0]
df['name'] = df.name.apply(return_model)
df.sample(5)

df.sort_values(by=['name'], ascending=True).head()



df['year'].describe()





df['owner'].describe()


df['owner'].unique()


df['owner'].value_counts()



df = df.drop(df[df.owner == 'Test Drive Car'].index)


df['owner'].value_counts() # проверка




