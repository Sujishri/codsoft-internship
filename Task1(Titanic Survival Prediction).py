#!/usr/bin/env python
# coding: utf-8

# # Importing required libraries

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


# # Loading datasets :

# In[20]:


df=pd.read_csv("Titanic-Dataset.csv")
df.head()


# # Data Preprocessing :

# In[21]:


df.info()


# In[25]:


df.isnull().sum()


# In[23]:


df.describe()


# # Handling missing values

# In[31]:


df['Age'].fillna(df['Age'].median() , inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0] , inplace=True)
df.drop(columns=['Cabin'], inplace=True)
df.drop(columns=['PassengerId','Name','Ticket'] ,inplace=True)
df=pd.get_dummies(df,columns=['Sex','Embarked'] , drop_first=True)


# # Visualizing the data

# In[37]:


plt.figure(figsize=(10,6))
sns.histplot(df['Age'],kde=True,bins=10)
plt.title('Age distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()


# In[39]:


plt.figure(figsize=(10,6))
sns.barplot(x='Pclass',y='Survived',data=df)
plt.title('Survival Rate by class')
plt.xlabel('Pclass')
plt.ylabel('Survived')
plt.show()


# In[43]:


plt.figure(figsize=(10,6))
sns.barplot(x='Sex_male',y='Survived',data=df)
plt.title('Survival rate by gender')
plt.xlabel('Gender')
plt.ylabel('Survived')
plt.show()


# In[50]:


plt.figure(figsize=(10,6))
sns.heatmap(df.corr(),annot=True,cmap='autumn',fmt='.2f')
plt.title('Correlation heatmap')
plt.show()


# In[56]:


sns.pairplot(df,hue='Survived',diag_kind='kde')
plt.show()


# # Training and Testing data:

# In[57]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
X=df.drop('Survived',axis=1)
y=df['Survived']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)


# # Using Logistic Regression

# In[64]:


from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
model=LogisticRegression(max_iter=1000)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)


# # Evaluating ML Model

# In[63]:


accu=accuracy_score(y_test,y_pred)
print(f'Accuracy: {accu:.2f}')

conf=confusion_matrix(y_test,y_pred)
print('Confusion Matrix:\n', conf)

cl_re=classification_report(y_test,y_pred)
print('Classification Report:\n', cl_re)


# In[ ]:




