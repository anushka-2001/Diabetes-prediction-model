#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df= pd.read_csv("C:\\Users\\Anushka\\Desktop\\diabetes.csv")


# In[4]:


df.head()


# In[5]:


df.info()


# In[6]:


df.isnull().any()


# In[31]:


corrmat = df.corr()
top_corr_features= corrmat.index
plt.figure(figsize=(20,20))
g=sns.heatmap(df[top_corr_features].corr(),annot = True,cmap="twilight")


# In[32]:


df.corr()


# In[9]:


true_count=len(df.loc[df['Outcome']==1])
false_count=len(df.loc[df['Outcome']==0])


# In[10]:


(true_count,false_count)


# In[13]:


from sklearn.model_selection import train_test_split
featured_columns = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
predicted_column=['Outcome']
X=df[featured_columns].values
Y=df[predicted_column].values
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size = 0.30,random_state=10)


# In[26]:


#from sklearn.impute import SimpleImputer
#imp = SimpleImputer(missing_values=np.nan, strategy='mean')
#X_train=fill_values.fit_transform(X_train)
#X_test = fill_values.fit_transform(X_test)


# In[27]:


from sklearn.ensemble import RandomForestClassifier
random_forest_model = RandomForestClassifier(random_state=10)
random_forest_model.fit(X_train,Y_train.ravel())
#ravel - This function returns a flattened one-dimensional array. A copy is made only if needed. The returned array will have the same type as that of the input array.
#The function takes one parameter


# In[28]:


predict_train_data = random_forest_model.predict(X_test)
from sklearn import metrics
print("Accuracy = {0:.3f}".format(metrics.accuracy_score(Y_test,predict_train_data)))


# In[ ]:
pickle.dump(regressor,open('model.pkl','wb'))
model = pickle.load(open())


