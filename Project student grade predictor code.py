#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


student=pd.read_csv('student-mat.csv')


# In[3]:


student.drop(['famsize', 'Mjob', 'Fjob', 'school','reason','guardian','nursery','higher'], axis=1, inplace=True)


# In[4]:


student.dtypes


# In[5]:


binary = ["sex", "Pstatus", "schoolsup", "famsup", "paid", "activities", "internet", "romantic"]


# In[6]:


def binary_encoder(dataset, col):
    dataset[col] = dataset[col].astype('category')
    dataset[col] = dataset[col].cat.codes
    dataset[col] = dataset[col].astype('int')


# In[7]:


for col in binary:
    binary_encoder(student, col)


# In[8]:


student.head()


# In[9]:


student.dtypes


# In[10]:


multiple=["address"]


# In[11]:


student = pd.get_dummies(student, columns=['address'])


# In[12]:


multiple2=["address_R","address_U"]


# In[13]:


for col in multiple2:
    binary_encoder(student, col)


# In[14]:


student.dtypes


# In[15]:


from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(student, test_size=0.25, random_state=42)


# In[16]:


student1=train_set.copy()


# In[17]:


corr_matrix=student1.corr()


# In[18]:


corr_matrix["G3"].sort_values(ascending=False)


# In[19]:


student1.plot(kind="scatter", x="G1", y="G3",
 alpha=0.8)


# In[20]:


prep=train_set.drop('G3',axis=1)


# In[21]:


student_labels=train_set['G3'].copy()


# In[22]:


from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(prep, student_labels)


# In[23]:


somedata = prep.iloc[:5]


# In[24]:


somedata_labels=student_labels.iloc[:5]


# In[25]:


print("Predictions:\t", lin_reg.predict(somedata))
print("Labels:\t\t", list(somedata_labels))



# In[26]:


from sklearn.metrics import mean_squared_error
student_predictions = lin_reg.predict(prep)
lin_mse = mean_squared_error(student_labels, student_predictions)
lin_rmse=np.sqrt(lin_mse)


# In[27]:


print(lin_rmse)


# In[29]:


## now checking for the test cases


# In[31]:


print("Predictions:\t", lin_reg.predict(student_test))
print("Labels:\t\t", list(student_label_test))


# In[32]:


student_test_predictions = lin_reg.predict(student_test)
lin_test_mse = mean_squared_error(student_label_test, student_test_predictions)
lin_test_rmse=np.sqrt(lin_test_mse)
print(lin_test_rmse)


# In[36]:


import pickle


# In[38]:


filename='student_grade_predictor'
pickle.dump(lin_reg,open(filename,'wb'))


# In[ ]:




