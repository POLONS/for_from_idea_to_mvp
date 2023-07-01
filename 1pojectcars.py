#!/usr/bin/env python
# coding: utf-8

# -------

#  #  <span style="color:darkblue"> Прогноз стоимости подержанного автомобиля </span>
#   ### <span style="color:darkblue"> Разработка ML сервиса: от идеи к прототипу </span>
#  by PT

# --------

# ![title](https://new-retail.ru/upload/iblock/388/388b7ba53f4e8e8bfab2f473ee8f7576.jpg)
# 

# <div style="text-align: right"> Copyright. new-retail.ru </div>

# ----------------

# ## <span style="color:darkblue">1. Общая информация </span>

# 
# 
#  *Признаки*
# * name / модель автомобиля
# * year / год выпуска с завода-изготовителя
# * km_driven / пробег на дату продажи
# * fuel / тип топлива
# * seller_type / продавец
# * transmission / тип трансмиссии
# * owner / какой по счёту хозяин
# * mileage / пробег
# * engine / рабочий объем двигателя
# * max_power / пиковая мощность двигателя
# * torque / крутящий момент
# * seats / количество мест

# In[588]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[589]:


df = pd.read_csv('cars.csv')


# <span style="color:darkblue">Первые 10 авто  из датасета: </span>

# In[590]:


df.head(11)


# In[591]:


df.sample(5)


# In[592]:


df.shape


# **<span style="color:darkblue">База из 6 999 автомобилей! </span>**

# In[593]:


df.info()


# In[594]:


df.describe() #for int,float only


# min year  - 1983, max - 2020.
# min seats - 2, max seats -  14.

# In[595]:


df.describe(include='object')


# Имеем датасет с пропусками по некоторым признакам. Так как пропущенных значнеий не много - *202 шт. =  2.9%*; почистим датасет от нулевых значений, так как сильного влияния на результат анализа такое изменение не окажет.  

# In[596]:


df.dropna(inplace=True)


# In[597]:


df.info()


# In[598]:


df.describe()


# Исчезли выбросы в значениях переменных 'year'( < 1994 - меньше 25%). 
# В selling_price и km_driven изменения незначительные. Для seats изменений нет.

# In[599]:


df.describe(include='object')


# Среди признаков типа object, после очистки нулевых значений, резких изменений по характеристикам нет.

# ----------

# ## <span style="color:darkblue">2. Однофакторный анализ </span>

# <span style="color:darkblue">Основные параметры влияющие на цену продажи  подержанного  автомобиля: \
#     Пробег на дату продажи, Трансмиссия, Пробег, Модель, Год выпуска, Количество владельцев.  [ https://avtocod.ru/; https://evaex.ru/ ]  </span>

# In[646]:


df.corr(method ='pearson')


# <span style="color:darkblue">Между числовыми параметрами коэффициент Пирсона показал прямую зависимость цены от года выпуска (>0,4) и обратную зависимость между  ценой и пробегом/пробегомна дату продажи (<-0,1).  </span>

# **<span style="color:darkblue">2.1 Пробег на дату продажи </span>**

# In[601]:


df[['km_driven']].describe().transpose()


# In[602]:


df[['km_driven']].sample(5)


# In[603]:


df[['km_driven']].max()


# In[604]:


sns.histplot(df['km_driven'], kde=True, legend=True)
plt.show()


# <span style="color:darkblue">Видим сильный разброс значений в km_driven.  **Чем меньше параметр, тем стоимость авто выше:** </span>

# In[605]:


df.sort_values(by=['km_driven'],ascending=True).plot.scatter('km_driven','selling_price');


# **<span style="color:darkblue">2.2 Трансмиссия </span>**

# In[606]:


df[['transmission', 'mileage']].describe().transpose()


# In[607]:


sns.histplot(df['transmission'])
plt.show()


# In[608]:


df['transmission'].value_counts()


# <span style="color:darkblue">Машин с механической коробкой больше в 6,5 раз (+658%) на  вторичном рынке, но их стоимость меньше. </span>

# In[609]:


df.sort_values(by=['transmission'],ascending=True).plot.scatter('transmission','selling_price');


# **<span style="color:darkblue">2.3 Пробег </span>**

# In[610]:


df[['mileage']].sample(5)


# Преобразуем mileage:

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


# Вид топлива влияет на единицы расчета пробега, поэтому  78 (= 1,15%) машин с более "зеленым" топливом не учитывались в дальнейшем анализе.

# In[614]:


df = df.drop(df[df.fuel == 'CNG'].index)


# In[615]:


df = df.drop(df[df.fuel == 'LPG'].index)


# In[616]:


df.describe() #изменения незначительные


# In[617]:


df[['mileage']].describe().transpose()


# **<span style="color:darkblue">75% пробега до 22,3 км! 90% в пределах 25 kmpl [км/л] </span>** 

# In[618]:


df['mileage'].quantile(0.9)


# In[619]:


df['mileage'].plot.box();


# In[620]:


df.sort_values(by=['mileage'],ascending=True).plot.scatter('mileage','selling_price');


# Естьзависимость между стоимостью и пробегом.

# **<span style="color:darkblue">2.4 Модель </span>**

# In[621]:


df['name'].describe().transpose()


# In[622]:


sns.histplot(df['name'])
plt.show()


# Анализ моделей с характеристиками не имеет смысла, поэтому  рассмотрим машины только по модели.

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


# **<span style="color:darkblue">Есть зависимость между ценой  и  маркой  автомобиля.</span>** 

# In[627]:


# Признак не используется в МО, так как очень большой набор значений. 
# Возможно разделение данных на кластеры и делать дальнейший прогноз по кластерам.


# **<span style="color:darkblue">2.5 Год выпуска </span>**

# In[628]:


df['year'].describe()


# In[629]:


sns.histplot(df['year'])
plt.show()


# In[630]:


df.sort_values(by=['year'],ascending=True).plot.scatter('year','selling_price');


#  **<span style="color:darkblue">Чем авто моложе, тем его стоимость выше.</span>** 

# In[631]:


# Categorial data - excluded
# Time series behave categorically different than data that is not ordered sequentially and we have to model them differently
#treat year as categorical variable and use some of the techniques such as One Hot Ecoding or Dummy Variables for better performance


# **<span style="color:darkblue">2.6 Количество владельцев </span>**

# In[632]:


df['owner'].describe()


# In[633]:


df['owner'].unique()


# In[634]:


df['owner'].value_counts()


# test drive car уберем из сета:

# In[635]:


df = df.drop(df[df.owner == 'Test Drive Car'].index)


# In[636]:


df['owner'].value_counts() # проверка


# In[637]:


sns.histplot(df['owner'])
plt.show()


# In[638]:


df.plot.scatter('owner','selling_price');


#  **<span style="color:darkblue">Обратная зависимость - цена растет с уменьшением количества владельцев.</span>** 

# -------

# ## <span style="color:darkblue">3. Машинное обучение </span>

# <span style="color:darkblue">Выбранные признаки отсортированы далее для выполнения предсказание цен на подержанные автомобили, путем множественной линейной регрессии:</span> 

# In[639]:


X = df.drop(['year','owner','name','selling_price', 'fuel', 'seller_type', 'engine', 'max_power','torque', 'seats'], axis=1)

y = df['selling_price'] # целевая переменная (target)


# In[640]:


X.head()


# Преобразуем данные в числовые:

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


#  **<span style="color:darkblue">35% of the variability in the outcome variable (y) can be explained by the predictor variables (x)</span>** 

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


#  **<span style="color:darkblue"> Результат: \
#  Точность модели  низкая -  35%. \
#  Параметры: km_driven, mileage, transmission не имеют сильного влияния на цену подержаного авто. \
# \
#  Рекомендации: \
#  Не использовать вышеперечисленные параметры для прогнозирования цен данного датасета. </span>**  

# ------
