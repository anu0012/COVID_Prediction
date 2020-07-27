
# coding: utf-8

# In[253]:


import os
import sys
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import plotly.express as px


# In[231]:


daily_data = pd.read_csv('data/daily.csv')
govt_measures = pd.read_excel('data/acaps_covid19_government_measures_dataset.xlsx', sheet_name='Database')
daily_tests = pd.read_csv('data/full-list-covid-19-tests-per-day.csv')
lockdown_data = pd.read_excel('data/lockdown_data.xlsx')
owid_data = pd.read_excel('data/owid-covid-data.xlsx')


# In[232]:


daily_data.sort_values('date',inplace=True)


# In[233]:


owid_data.columns


# In[234]:


# Convert date to datetime object 
daily_data['date'] = pd.to_datetime(daily_data['date'], format='%Y%m%d')
daily_tests['Date'] = pd.to_datetime(daily_tests['Date'], format='%b %d, %Y')
owid_data['date'] = pd.to_datetime(owid_data['date'], format='%Y-%m-%d')


# In[235]:


# Filter only for United States
govt_measures = govt_measures[govt_measures['COUNTRY'] == 'United States of America']
daily_tests = daily_tests[daily_tests['Entity'] == 'United States']
owid_data = owid_data[owid_data['location'] == 'United States']


# In[236]:


#daily_data.merge(govt_measures, left_on='date', right_on='DATE_IMPLEMENTED', how='left')


# In[237]:


daily_data = daily_data.merge(daily_tests, left_on='date', right_on='Date', how='left')
daily_data = daily_data.merge(lockdown_data, on='date', how='left')
daily_data = daily_data.merge(owid_data[['date','stringency_index']], on='date', how='left')


# In[238]:


daily_data.columns


# In[239]:


feature_names = ['date','positive','totalTestResultsIncrease','is_lockdown','stringency_index']
daily_data = daily_data[feature_names]


# In[240]:


x = daily_data.copy()
x.index = x.date
x.drop('date',axis=1,inplace=True)


# In[241]:


train = x[:int(0.8*(len(x)))]
valid = x[int(0.8*(len(x))):]


# ### Model 1 (VAR)

# In[242]:


from statsmodels.tsa.vector_ar.var_model import VAR

model = VAR(endog=train)
model_fit = model.fit()


# In[243]:


prediction = model_fit.forecast(model_fit.y, steps=len(valid))


# In[244]:


pred = pd.DataFrame(index=range(0,len(prediction)),columns=[cols])
for j in range(0,2):
    for i in range(0, len(prediction)):
       pred.iloc[i][j] = prediction[i][j]

#check rmse
#for i in cols:
print('rmse value for Positive cases is : ', math.sqrt(mean_squared_error(pred['positive'], valid['positive'])))


# In[245]:


model = VAR(endog=x)
model_fit = model.fit()
yhat = model_fit.forecast(model_fit.y, steps=30)
print(yhat)


# In[246]:


forecast = []
for i in yhat:
    forecast.append(i[0])


# In[247]:


pd.Series(forecast).plot()


# In[248]:


pd.date_range(start='25/7/2020', periods=30)


# In[249]:


# result output
result = pd.DataFrame()
result['date'] = pd.date_range(start='25/7/2020', periods=30)
result['number_of_positives'] = forecast


# In[250]:


result = result[(result['date'] >= '27/7/2020') & (result['date'] <= '15/8/2020')]
result.to_csv('result.csv',index=False)


# In[254]:


fig = px.line(result, x='date', y='number_of_positives')
fig.show()


# ### Model 2 (ARIMAX)

# In[251]:


from statsmodels.tsa.arima_model import ARIMA

model = ARIMA(train.positive, exog=train.totalTestResultsIncrease, order=(1,1,1))
model_fit = model.fit()


# In[252]:


model_fit.summary()

