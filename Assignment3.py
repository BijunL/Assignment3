#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt


# # 0.) Clean the Apple Data to get a quarterly series of EPS.

# In[2]:


y = pd.read_csv('AAPL_quarterly_financials.csv')


# In[3]:


y.index = y.name


# In[4]:


y = pd.DataFrame(y.loc["BasicEPS",:]).iloc[2:,:]


# In[5]:


y.index = pd.to_datetime(y.index)


# In[6]:


y = y.fillna(0.).sort_index()


# # 2.) Normalize all the X data

# In[7]:


from pytrends.request import TrendReq


# In[9]:


# Create pytrends object
pytrends = TrendReq(hl='en-US', tz=360)

# Set up the keywords and the timeframe
keywords = ["MacBook","iPhone","iPad","Apple Layoffs","Sales","Apple Share Price",
            "Recession","Policy","Taylor Swift Tickets","Is the Earth Flat","Hospital"]  # Add your keywords here
start_date = '2004-01-01'
end_date = '2024-01-01'

# Create an empty DataFrame to store the results
df = pd.DataFrame()

# Iterate through keywords and fetch data
for keyword in keywords:
    pytrends.build_payload([keyword], cat=0, timeframe=f'{start_date} {end_date}', geo='', gprop='')
    interest_over_time_df = pytrends.interest_over_time()
    df[keyword] = interest_over_time_df[keyword]


# In[10]:


X = df.resample("Q").mean()
temp = pd.concat([y,X],axis=1).dropna()
y = temp[["BasicEPS"]].copy()
X = temp.iloc[:,1:].copy()


# In[11]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# # 4.) Run a Lasso with lambda of .5. Plot a bar chart.

# In[12]:


from sklearn.linear_model import Lasso
lasso = Lasso(alpha = .1)
lasso.fit(X_scaled,y)
coefficients = lasso.coef_


# In[13]:


plt.figure(figsize = (18,5))
plt.bar(range(len(coefficients)),coefficients,tick_label=X.columns)
plt.axhline(0.,color = "red")
plt.show()


# # 5.) Do these coefficient magnitudes make sense?

# In[ ]:





# In[ ]:





# # 6.) Run a for loop looking at 10 different Lambdas and plot the coefficient magnitude for each.

# In[ ]:





# In[ ]:





# # 7.) Run a cross validation. What is your ideal lambda?

# In[ ]:





# In[ ]:




