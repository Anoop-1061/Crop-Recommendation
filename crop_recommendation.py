#!/usr/bin/env python
# coding: utf-8

# ## Data fields
# ### N - ratio of Nitrogen content in soil - kg/ha
# ### P - ratio of Phosphorous content in soil - kg/ha
# ### K - ratio of Potassium content in soil - kg/ha
# ### temperature - temperature in degree Celsius
# ### humidity - relative humidity in %
# ### ph - ph value of the soil
# ### rainfall - rainfall in mm

# ## Import Libraries

# In[1]:


import pandas as pd
import numpy as np
import random

# for interactivity
import ipywidgets
from ipywidgets import interact

from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=False)
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots


# In[2]:


crop = pd.read_csv("Crop_recommendation.csv")
crop.head()


# In[3]:


crop.shape


# In[4]:


crop.columns


# In[5]:


crop.isnull().any()


# ## Descriptive Statistics

# In[6]:


crop.describe().transpose()


# In[7]:


print("Number of various crops: ", len(crop['label'].unique()))
print("List of crops: ", crop['label'].unique())


# In[8]:


crop['label'].value_counts()


# In[9]:


# lets check the Summary for all the crops

print("Average Ratio of Nitrogen in the Soil : {0:.2f}".format(crop['N'].mean()))
print("Average Ratio of Phosphorous in the Soil : {0:.2f}".format(crop['P'].mean()))
print("Average Ratio of Potassium in the Soil : {0:.2f}".format(crop['K'].mean()))
print("Average Tempature in Celsius : {0:.2f}".format(crop['temperature'].mean()))
print("Average Relative Humidity in % : {0:.2f}".format(crop['humidity'].mean()))
print("Average PH Value of the soil : {0:.2f}".format(crop['ph'].mean()))
print("Average Rainfall in mm : {0:.2f}".format(crop['rainfall'].mean()))


# In[10]:


## Lets compare the Average Requirement for each crops with average conditions

@interact
def compare(conditions = ['N','P','K','temperature','ph','humidity','rainfall']):
    print("Average Value for", conditions,"is {0:.2f}".format(crop[conditions].mean()))
    print("----------------------------------------------")
    print("Rice : {0:.2f}".format(crop[(crop['label'] == 'rice')][conditions].mean()))
    print("Black Grams : {0:.2f}".format(crop[crop['label'] == 'blackgram'][conditions].mean()))
    print("Banana : {0:.2f}".format(crop[(crop['label'] == 'banana')][conditions].mean()))
    print("Jute : {0:.2f}".format(crop[crop['label'] == 'jute'][conditions].mean()))
    print("Coconut : {0:.2f}".format(crop[(crop['label'] == 'coconut')][conditions].mean()))
    print("Apple : {0:.2f}".format(crop[crop['label'] == 'apple'][conditions].mean()))
    print("Papaya : {0:.2f}".format(crop[(crop['label'] == 'papaya')][conditions].mean()))
    print("Muskmelon : {0:.2f}".format(crop[crop['label'] == 'muskmelon'][conditions].mean()))
    print("Grapes : {0:.2f}".format(crop[(crop['label'] == 'grapes')][conditions].mean()))
    print("Watermelon : {0:.2f}".format(crop[crop['label'] == 'watermelon'][conditions].mean()))
    print("Kidney Beans: {0:.2f}".format(crop[(crop['label'] == 'kidneybeans')][conditions].mean()))
    print("Mung Beans : {0:.2f}".format(crop[crop['label'] == 'mungbean'][conditions].mean()))
    print("Oranges : {0:.2f}".format(crop[(crop['label'] == 'orange')][conditions].mean()))
    print("Chick Peas : {0:.2f}".format(crop[crop['label'] == 'chickpea'][conditions].mean()))
    print("Lentils : {0:.2f}".format(crop[(crop['label'] == 'lentil')][conditions].mean()))
    print("Cotton : {0:.2f}".format(crop[crop['label'] == 'cotton'][conditions].mean()))
    print("Maize : {0:.2f}".format(crop[(crop['label'] == 'maize')][conditions].mean()))
    print("Moth Beans : {0:.2f}".format(crop[crop['label'] == 'mothbeans'][conditions].mean()))
    print("Pigeon Peas : {0:.2f}".format(crop[(crop['label'] == 'pigeonpeas')][conditions].mean()))
    print("Mango : {0:.2f}".format(crop[crop['label'] == 'mango'][conditions].mean()))
    print("Pomegranate : {0:.2f}".format(crop[(crop['label'] == 'pomegranate')][conditions].mean()))
    print("Coffee : {0:.2f}".format(crop[crop['label'] == 'coffee'][conditions].mean()))


# In[11]:


# lets check the Summary Statistics for each of the Crops

@interact
def summary(crops = list(crop['label'].value_counts().index)):
    x = crop[crop['label'] == crops]
    print("---------------------------------------------")
    print("Statistics for Nitrogen")
    print("Minimum Nitrigen required :", x['N'].min())
    print("Average Nitrogen required :", x['N'].mean())
    print("Maximum Nitrogen required :", x['N'].max()) 
    print("---------------------------------------------")
    print("Statistics for Phosphorous")
    print("Minimum Phosphorous required :", x['P'].min())
    print("Average Phosphorous required :", x['P'].mean())
    print("Maximum Phosphorous required :", x['P'].max()) 
    print("---------------------------------------------")
    print("Statistics for Potassium")
    print("Minimum Potassium required :", x['K'].min())
    print("Average Potassium required :", x['K'].mean())
    print("Maximum Potassium required :", x['K'].max()) 
    print("---------------------------------------------")
    print("Statistics for Temperature")
    print("Minimum Temperature required : {0:.2f}".format(x['temperature'].min()))
    print("Average Temperature required : {0:.2f}".format(x['temperature'].mean()))
    print("Maximum Temperature required : {0:.2f}".format(x['temperature'].max()))
    print("---------------------------------------------")
    print("Statistics for Humidity")
    print("Minimum Humidity required : {0:.2f}".format(x['humidity'].min()))
    print("Average Humidity required : {0:.2f}".format(x['humidity'].mean()))
    print("Maximum Humidity required : {0:.2f}".format(x['humidity'].max()))
    print("---------------------------------------------")
    print("Statistics for PH")
    print("Minimum PH required : {0:.2f}".format(x['ph'].min()))
    print("Average PH required : {0:.2f}".format(x['ph'].mean()))
    print("Maximum PH required : {0:.2f}".format(x['ph'].max()))
    print("---------------------------------------------")
    print("Statistics for Rainfall")
    print("Minimum Rainfall required : {0:.2f}".format(x['rainfall'].min()))
    print("Average Rainfall required : {0:.2f}".format(x['rainfall'].mean()))
    print("Maximum Rainfall required : {0:.2f}".format(x['rainfall'].max()))


# In[12]:


# lets make this funtion more Intuitive

@interact
def compare(conditions = ['N','P','K','temperature','ph','humidity','rainfall']):
    print("Crops which require greater than average", conditions,'\n')
    print(crop[crop[conditions] > crop[conditions].mean()]['label'].unique())
    print("----------------------------------------------")
    print("Crops which require less than average", conditions,'\n')
    print(crop[crop[conditions] <= crop[conditions].mean()]['label'].unique())


# ## Data Visualization and analysis

# In[13]:


crop_summary = pd.pivot_table(crop,index=['label'],aggfunc='mean')
crop_summary.head()


# In[14]:


colorarr = ['#0592D0','#Cd7f32', '#E97451', '#Bdb76b', '#954535', '#C2b280', '#808000','#C2b280', '#E4d008', '#9acd32', '#Eedc82', '#E4d96f',
           '#32cd32','#39ff14','#00ff7f', '#008080', '#36454f', '#F88379', '#Ff4500', '#Ffb347', '#A94064', '#E75480', '#Ffb6c1', '#E5e4e2',
           '#Faf0e6', '#8c92ac', '#Dbd7d2','#A7a6ba', '#B38b6d']


# #### Nitrogen Analysis

# In[15]:


crop_summary_N = crop_summary.sort_values(by='N', ascending=False)
  
fig = make_subplots(rows=1, cols=1)

top = {
    'y' : crop_summary_N['N'][0:22].sort_values().index,
    'x' : crop_summary_N['N'][0:22].sort_values()
}

fig.add_trace(
    go.Bar(top,
           name="Nitrogen required",
           marker_color=random.choice(colorarr),
           orientation='h',
          text=top['x']),
    
    row=1, col=1
)

fig.update_traces(texttemplate='%{text}', textposition='inside')
fig.update_layout(title_text="Nitrogen (N)",
                  plot_bgcolor='white',
                  font_size=12, 
                  font_color='black',
                 height=700)

fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.show()


# #### Phosphorous Analysis

# In[16]:


crop_summary_P = crop_summary.sort_values(by='P', ascending=False)
  
fig = make_subplots(rows=1, cols=1)

top = {
    'y' : crop_summary_P['P'][0:22].sort_values().index,
    'x' : crop_summary_P['P'][0:22].sort_values()
}

fig.add_trace(
    go.Bar(top,
           name="Phosphorus  required",
           marker_color=random.choice(colorarr),
           orientation='h',
          text=top['x']),
    
    row=1, col=1
)

fig.update_traces(texttemplate='%{text}', textposition='inside')
fig.update_layout(title_text="Phosphorus (P)",
                  plot_bgcolor='white',
                  font_size=12, 
                  font_color='black',
                 height=700)

fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.show()


# #### Potassium Analysis

# In[17]:


crop_summary_K = crop_summary.sort_values(by='K', ascending=False)
  
fig = make_subplots(rows=1, cols=1)

top = {
    'y' : crop_summary_K['N'][0:22].sort_values().index,
    'x' : crop_summary_K['K'][0:22].sort_values()
}

fig.add_trace(
    go.Bar(top,
           name="Potassium required",
           marker_color=random.choice(colorarr),
           orientation='h',
          text=top['x']),
    
    row=1, col=1
)

fig.update_traces(texttemplate='%{text}', textposition='inside')
fig.update_layout(title_text="Potassium (K)",
                  plot_bgcolor='white',
                  font_size=12, 
                  font_color='black',
                 height=700)

fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.show()


# #### NPK Analysis

# In[18]:


fig = go.Figure()
fig.add_trace(go.Bar(
    x=crop_summary.index,
    y=crop_summary['N'],
    name='Nitrogen',
    marker_color='mediumvioletred'
))
fig.add_trace(go.Bar(
    x=crop_summary.index,
    y=crop_summary['P'],
    name='Phosphorous',
    marker_color='springgreen'
))
fig.add_trace(go.Bar(
    x=crop_summary.index,
    y=crop_summary['K'],
    name='Potash',
    marker_color='dodgerblue'
))

fig.update_layout(title="N-P-K values comparision between crops",
                  plot_bgcolor='white',
                  barmode='group',
                  xaxis_tickangle=-45,
                 width = 750)

fig.show()


# In[19]:


import seaborn as sns
sns.heatmap(crop.corr(), annot=True)


# ## Declare independent and target variables

# In[20]:


X = crop.drop('label', axis=1)
y = crop['label']


# ## Encoding the target

# In[21]:


from sklearn import preprocessing as pre
LE = pre.LabelEncoder()
y = LE.fit_transform(y)
y


# In[22]:


np.unique(y)


# In[23]:


# Saving the encoder for inverse transforming the predictions.
import pickle
pickle.dump(LE, open('label.pkl', 'wb'))


# In[24]:


# Initializing empty lists to append all model's name and corresponding name
acc = []
model = []


# ## Split dataset into training and test set

# In[25]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
                                                    shuffle = True, random_state = 0)


# In[26]:


print('Train data shape:', (X_train.shape))
print('Test data shape:', (X_test.shape))


# ## Scaling the Features

# In[27]:


# Robust sacler is used as there may be some outliers in the data.
scaler = pre.RobustScaler()
# Train data
#train_scaled = scaler.fit_transform(X_train)
scaler.fit(X_train)
train_scaled = scaler.transform(X_train)
train_scaled


# In[28]:


# Test data
test_scaled = scaler.transform(X_test)
test_scaled


# In[29]:


# Saving the scaler for scaling the inputs for prediction.
pickle.dump(scaler, open('scaler.pkl', 'wb'))


# # Preparing Machine Learning Models for the Splitted Dataset

# ## 1. Decision Tree

# In[30]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import tree
from __future__ import print_function
import warnings
warnings.filterwarnings('ignore')

DecisionTree = DecisionTreeClassifier(criterion="entropy",random_state=2,max_depth=5)

DecisionTree.fit(train_scaled, y_train)

predicted_values = DecisionTree.predict(test_scaled)
x = metrics.accuracy_score(y_test, predicted_values)
acc.append(x)
model.append('Decision Tree')
print("DecisionTrees's Accuracy is: ", x*100)

print(classification_report(y_test,predicted_values))


# In[31]:


from sklearn.model_selection import cross_val_score


# In[32]:


# Cross validation score (Decision Tree)
score = cross_val_score(DecisionTree, X, y,cv=5)
score


# ## 2. Gaussian Naive Bayes

# In[33]:


from sklearn.naive_bayes import GaussianNB

NaiveBayes = GaussianNB()

NaiveBayes.fit(train_scaled,y_train)

predicted_values = NaiveBayes.predict(test_scaled)
x = metrics.accuracy_score(y_test, predicted_values)
acc.append(x)
model.append('Naive Bayes')
print("Naive Bayes's Accuracy is: ", x*100)

print(classification_report(y_test,predicted_values))


# In[34]:


# Cross validation score (NaiveBayes)
score = cross_val_score(NaiveBayes,X,y,cv=5)
score


# ## 3. Support Vector Machine 

# In[35]:


from sklearn.svm import SVC

SVM = SVC(gamma='auto')

SVM.fit(train_scaled,y_train)

predicted_values = SVM.predict(test_scaled)

x = metrics.accuracy_score(y_test, predicted_values)
acc.append(x)
model.append('SVM')
print("SVM's Accuracy is: ", x*100)

print(classification_report(y_test,predicted_values))


# In[36]:


# Cross validation score (SVM)
score = cross_val_score(SVM,X,y,cv=5)
score


# ## 4. Logistic Regression

# In[37]:


from sklearn.linear_model import LogisticRegression

LogReg = LogisticRegression(random_state=2)

LogReg.fit(train_scaled,y_train)

predicted_values = LogReg.predict(test_scaled)

x = metrics.accuracy_score(y_test, predicted_values)
acc.append(x)
model.append('Logistic Regression')
print("Logistic Regression's Accuracy is: ", x*100)

print(classification_report(y_test,predicted_values))


# In[38]:


# Cross validation score (Logistic Regression)
score = cross_val_score(LogReg,X,y,cv=5)
score


# ## 5. Random Forest

# In[39]:


from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier(n_estimators=20, random_state=0)
RF.fit(train_scaled,y_train)

predicted_values = RF.predict(test_scaled)

x = metrics.accuracy_score(y_test, predicted_values)
acc.append(x)
model.append('RF')
print("RF's Accuracy is: ", x*100)

print(classification_report(y_test,predicted_values))


# In[40]:


# Cross validation score (Random Forest)
score = cross_val_score(RF,X,y,cv=5)
score


# ## 6. K Nearest Neighbour

# In[41]:


from sklearn.neighbors import KNeighborsClassifier


# In[42]:


KNN = KNeighborsClassifier()

KNN.fit(train_scaled,y_train)

predicted_values = KNN.predict(test_scaled)
x = metrics.accuracy_score(y_test, predicted_values)
acc.append(x)
model.append('K-Nearest-Neighbour')
print("KNN's Accuracy is: ", x*100)

print(classification_report(y_test,predicted_values))


# In[43]:


# Cross validation score (Random Forest)
score = cross_val_score(KNN,X,y,cv=5)
score


# ## Accuracy comparison of different Machine Learning Models

# In[44]:


import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8, 10))
sns.barplot(y=acc, x=model, capsize=0.2, ax=ax)

# show the mean
for p in ax.patches:
    h, w, x = p.get_height(), p.get_width(), p.get_x()
    xy = (x + w / 2., h / 2)
    text = f':{h:0.2f}'
    ax.annotate(text=text, xy=xy, ha='center', va='center')

ax.set(xlabel='Accuracy', ylabel='Algorithm')
plt.show()


# ## Saving the best Machine Learning Model i.e Random Forest model

# In[45]:


import pickle
# Dump the trained Naive Bayes classifier with Pickle
pkl_filename = 'RF.pkl'
# Open the file to save as pkl file
Model_pkl = open(pkl_filename, 'wb')
pickle.dump(RF, Model_pkl)
# Close the pickle instances
Model_pkl.close()

