#!/usr/bin/env python
# coding: utf-8

# <a href="https://drive.google.com/drive/folders/176VLvhLVMz4-ZFKh1uHpCijZJfQWd-6w?usp=sharing">Link to the dataset</a>
# 

# <blockquote>
# Utility function declaration
# </blockquote>

# In[1]:


# input: Pandas DataFrame named data
# output: print a summary of the passing Pandas DataFrame 
def printSummary(data):
    print('Number of objects = {}'.format(data.shape[0]))
    print('Number of attributes = {}'.format(data.shape[1]))
    print('   | Column                      | Missing values | Infinity values')
    print('-------------------------------------------------------------------')
    i = 0
    for col_label in data.columns:
        print('{0:2d} | {1:27s} | {2:14d} | {3:15d}'.format(i, col_label, data[col_label].isnull().sum(), data[col_label].isin([np.inf]).sum()))
        i = i + 1
    print('-------------------------------------------------------------------')


# # 1. Data cleaning 

# In[2]:


import pandas as pd
import numpy as np
import warnings
# suppress the warning caused by setting the first column as index column
warnings.simplefilter(action='ignore', category=FutureWarning) 
data = pd.read_csv('cic2017-ddos-data.csv', index_col=0, header=0)
# dataset contains infinite values in some columns
data = data.replace('inf', np.inf)

print('Dataset before cleaning:')
printSummary(data)
print('Label column information:')
print(data.loc[:, 'Label'].describe())


# In[3]:


data = data.replace(np.nan, np.inf)
data['Flow Bytes/s'].replace(np.inf, data['Flow Bytes/s'].median(), inplace=True)
data['Flow Packets/s'].replace(np.inf, data['Flow Packets/s'].median(), inplace=True)

print('Dataset after cleaning:')
printSummary(data)


# # 2. Data preprocessing

# In[4]:


print('Number of matching values of two columns \"Fwd Header Length\" and \"Fwd Header Length - dupl\": {}'.format(data['Fwd Header Length'].eq(data['Fwd Header Length - dupl']).sum()))
# drop two duplicated columns
data = data.drop(['Fwd Header Length - dupl'], axis=1)
print('Dataset after dropping duplicated column: ')
print('Number of instances = {}'.format(data.shape[0]))
print('Number of attributes = {}'.format(data.shape[1]))


# In[5]:


from sklearn.decomposition import PCA

numInstances = data.shape[0]
numComponents = 10
pca = PCA(n_components=numComponents)
pca.fit(data.loc[:, 'Flow Duration':'Idle Min'])

projected = pca.transform(data.loc[:, 'Flow Duration':'Idle Min'])
projected = pd.DataFrame(projected,columns=['PC1','PC2','PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10'], index=range(numInstances))
projected.head(20)


# In[6]:


from sklearn.model_selection import train_test_split

new = pd.concat([data.iloc[:, 0:7], projected, data.iloc[:, 83]], axis=1)
print('Dataset after performing PCA:')
printSummary(new)

y = new.loc[:, 'Label']
x = new.drop(['Label'], axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2)
print('Number of objects in training set: {}'.format(x_train.shape[0]))
print('Number of objects in testing set:  {}'.format(x_test.shape[0]))


# # 3. Data mining
# ## Decision tree

# ### Decision tree max depth evaluation

# In[7]:


import time
from sklearn import tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plot

t_start = time.time_ns()
maxDepth = np.arange(1, 51)
trainAccuracy = np.zeros(50)
testAccuracy = np.zeros(50)
for i in range(1, 51):    
    clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=maxDepth[i - 1])
    clf = clf.fit(x_train.loc[:, 'PC1':'PC10'], y_train)

    y_pred_train = clf.predict(x_train.loc[:, 'PC1':'PC10'])
    trainAccuracy[i - 1] = accuracy_score(y_train, y_pred_train)
    y_pred = clf.predict(x_test.loc[:, 'PC1':'PC10'])
    testAccuracy[i - 1] = accuracy_score(y_test, y_pred)

plot.plot(maxDepth, trainAccuracy,'ro-', maxDepth,testAccuracy,'bv--')
plot.legend(['Training Accuracy', 'Test Accuracy'])
plot.xlabel('Max depth')
plot.ylabel('Accuracy')  
duration = time.time_ns() - t_start
print('Time to find max depth in nanosecond: {}'.format(duration))


# ### Decision tree applying on data

# In[8]:


t_start = time.time_ns()
maxDepth = 5
clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=maxDepth)
clf = clf.fit(x_train.loc[:, 'PC1':'PC10'], y_train)
duration = time.time_ns() - t_start
print('Time to build the Decision Tree classifier in nanosecond: {}'.format(duration))

y_pred_train = clf.predict(x_train.loc[:, 'PC1':'PC10'])
t_start = time.time_ns()
y_pred = clf.predict(x_test.loc[:, 'PC1':'PC10'])
duration = time.time_ns() - t_start
print('Time to run on the test set in nanosecond: {}'.format(duration))

print('Decision tree with max depth = {}'.format(maxDepth))
print('Accuracy on training data is {}'.format(accuracy_score(y_train, y_pred_train)))
print('Accuracy on testing data is  {}'.format(accuracy_score(y_test, y_pred)))    


# ### Decision tree classifier

# In[9]:


import pydotplus 
from IPython.display import Image

dot_data = tree.export_graphviz(clf, feature_names=new.loc[:, 'PC1':'PC10'].columns, class_names=['BENIGN','DDoS'], filled=True, 
                                out_file=None) 
graph = pydotplus.graph_from_dot_data(dot_data) 
Image(graph.create_png())


# ## K-nearest neighbors

# ### Hyperparameter k tunning

# In[10]:


from sklearn.neighbors import KNeighborsClassifier

t_start = time.time_ns()
k = np.arange(1, 51)
trainAccuracy = np.zeros(50)
testAccuracy = np.zeros(50)
for i in range(1, 51):    
    clf = KNeighborsClassifier(n_neighbors=i, metric='minkowski', p=2)
    clf = clf.fit(x_train.loc[:, 'PC1':'PC10'], y_train)

    y_pred_train = clf.predict(x_train.loc[:, 'PC1':'PC10'])
    trainAccuracy[i - 1] = accuracy_score(y_train, y_pred_train)
    y_pred = clf.predict(x_test.loc[:, 'PC1':'PC10'])
    testAccuracy[i - 1] = accuracy_score(y_test, y_pred)

plot.plot(k, trainAccuracy,'ro-', k ,testAccuracy,'bv--')
plot.legend(['Training Accuracy', 'Test Accuracy'])
plot.xlabel('Value of k')
plot.ylabel('Accuracy')   
duration = time.time_ns() - t_start
print('Time to find k in nanosecond: {}'.format(duration))


# ### K-nearest neighbor applying on data

# In[11]:


t_start = time.time_ns()
k = 3
clf = KNeighborsClassifier(n_neighbors=k, metric='minkowski', p=2)
clf = clf.fit(x_train.loc[:, 'PC1':'PC10'], y_train)
duration = time.time_ns() - t_start
print('Time to build the K-nearest Neighbor classifier in nanosecond: {}'.format(duration))

y_pred_train = clf.predict(x_train.loc[:, 'PC1':'PC10'])
t_start = time.time_ns()
y_pred = clf.predict(x_test.loc[:, 'PC1':'PC10'])
duration = time.time_ns() - t_start
print('Time to run on the test set in nanosecond: {}'.format(duration))


print('K-nearest neighbors with k = {}'.format(k))
print('Accuracy on training data is {}'.format(accuracy_score(y_train, y_pred_train)))
print('Accuracy on testing data is  {}'.format(accuracy_score(y_test, y_pred)))    


# ## Artificial Neural Network: Perceptron

# ### With PCA

# In[12]:


from sklearn.linear_model import Perceptron

# run perceptron with PCA
t_start = time.time_ns()
clf = Perceptron(tol=1e-3, fit_intercept=False, random_state=2)
clf = clf.fit(x_train.loc[:, 'PC1':'PC10'], y_train)
duration = time.time_ns() - t_start
print('Time to build the Perceptron classifier (on PCA dataset) in nanosecond: {}'.format(duration))

y_pred_train = clf.predict(x_train.loc[:, 'PC1':'PC10'])
t_start = time.time_ns()
y_pred = clf.predict(x_test.loc[:, 'PC1':'PC10'])
duration = time.time_ns() - t_start
print('Time to run on the test set in nanosecond: {}'.format(duration))

print('Perceptron with PCA:')
print('Accuracy on training data is {}'.format(accuracy_score(y_train, y_pred_train)))
print('Accuracy on testing data is  {}'.format(accuracy_score(y_test, y_pred)))    


# ### Without PCA

# In[13]:


# run perceptron without PCA
y_original = data.loc[:, 'Label']
x_original = data.drop(['Label'], axis=1)
x_train, x_test, y_train, y_test = train_test_split(x_original, y_original, test_size=0.3, random_state=2)

t_start = time.time_ns()
clf = Perceptron(tol=1e-3, random_state=2)
clf = clf.fit(x_train.loc[:, 'Flow Duration':'Idle Min'], y_train)
duration = time.time_ns() - t_start
print('Time to build the Perceptron classifier (on original dataset) in nanosecond: {}'.format(duration))

y_pred_train = clf.predict(x_train.loc[:, 'Flow Duration':'Idle Min'])
t_start = time.time_ns()
y_pred = clf.predict(x_test.loc[:, 'Flow Duration':'Idle Min'])
duration = time.time_ns() - t_start
print('Time to run on the test set in nanosecond: {}'.format(duration))

print('Perceptron without PCA:')
print('Accuracy on training data is {}'.format(accuracy_score(y_train, y_pred_train)))
print('Accuracy on testing data is  {}'.format(accuracy_score(y_test, y_pred)))    


# ### Perceptron accuracy visualization (with PCA)

# In[14]:


from sklearn.linear_model import SGDClassifier

y = new.loc[:, 'Label']
x = new.drop(['Label'], axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2)

lrate = np.arange(0.1, 2.1, 0.1)
trainAccuracy = np.zeros(20)
testAccuracy = np.zeros(20)
for i in range(20):
    clf = SGDClassifier(loss="perceptron", fit_intercept=False, eta0=lrate[i-1], learning_rate="constant", penalty=None)
    clf = clf.fit(x_train.loc[:, 'PC1':'PC10'], y_train)

    y_pred_train = clf.predict(x_train.loc[:, 'PC1':'PC10'])
    trainAccuracy[i - 1] = accuracy_score(y_train, y_pred_train)
    y_pred = clf.predict(x_test.loc[:, 'PC1':'PC10'])
    testAccuracy[i - 1] = accuracy_score(y_test, y_pred)

plot.plot(lrate, trainAccuracy,'ro-', lrate, testAccuracy,'bv--')
plot.legend(['Training Accuracy', 'Test Accuracy'])
plot.xlabel('Learning rate')
plot.ylabel('Accuracy') 


# # 4. Data visualization

# In[15]:


# stratification subsetting the dataset to visualize
# x, y are results from the dataset that was applied PCA
x_taken, x_left, y_taken, y_left = train_test_split(x, y, test_size=0.5, random_state=2, stratify=y)
x_taken = x_taken.drop(['Flow ID', 'Source IP', 'Source Port', 'Destination IP', 'Destination Port', 'Protocol', 'Timestamp'], axis=1)
visualizedData = pd.concat([x_taken, y_taken], axis=1)
print('Number of objects in the subset to be visualized: {}'.format(visualizedData.shape[0]))
visualizedData.head(20)


# In[16]:


# Andrews' curves plot
pd.plotting.andrews_curves(visualizedData, 'Label', color=('#556270', '#C7F464'))

