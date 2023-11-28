#!/usr/bin/env python
# coding: utf-8

# # Breast Cancer Classification

# ## Attribute Information:
# 
# -  ID number 
# - Diagnosis (M = malignant, B = benign)
# 
# ### Ten real-valued features are computed for each cell nucleus:
# 
# - radius (mean of distances from center to points on the perimeter)
# - texture (standard deviation of gray-scale values)
# - perimeter
# - area
# - smoothness (local variation in radius lengths)
# - compactness (perimeter^2 / area - 1.0)
# - concavity (severity of concave portions of the contour)
# - concave points (number of concave portions of the contour)
# - symmetry
# - fractal dimension ("coastline approximation" - 1)

# ##  Importing libraries

# In[1]:


# Importing libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno

import warnings
warnings.filterwarnings('ignore')


plt.style.use('ggplot')


# ## Load the data

# In[2]:


df = pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')


# In[3]:


df.head()


# ## Data Preprocessing

# In[4]:


df.drop(['id', 'Unnamed: 32'], axis = 1, inplace = True)


# In[5]:


df.diagnosis.unique()


# In[6]:


df['diagnosis'] = df['diagnosis'].apply(lambda val: 1 if val == 'M' else 0)


# In[7]:


df.head()


# In[8]:


df.describe()


# In[9]:


df.info()


# In[10]:


# checking for null values

df.isna().sum()


# In[11]:


# visualizing null values

msno.bar(df)


# #### There are no missing values in the data.

# ## Exploratory Data Analysis (EDA)

# In[12]:


plt.figure(figsize = (20, 15))
plotnumber = 1

for column in df:
    if plotnumber <= 30:
        ax = plt.subplot(5, 6, plotnumber)
        sns.distplot(df[column])
        plt.xlabel(column)
        
    plotnumber += 1

plt.tight_layout()
plt.show()


# In[13]:


# heatmap 

plt.figure(figsize = (20, 12))

corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype = bool))

sns.heatmap(corr, mask = mask, linewidths = 1, annot = True, fmt = ".2f")
plt.show()


# ### We can see that there are many columns which are very highly correlated which causes multicollinearity so we have to remove highly correlated features.

# In[14]:


# removing highly correlated features

corr_matrix = df.corr().abs() 

mask = np.triu(np.ones_like(corr_matrix, dtype = bool))
tri_df = corr_matrix.mask(mask)

to_drop = [x for x in tri_df.columns if any(tri_df[x] > 0.92)]

df = df.drop(to_drop, axis = 1)

print(f"The reduced dataframe has {df.shape[1]} columns.")


# In[15]:


# creating features and label 

X = df.drop('diagnosis', axis = 1)
y = df['diagnosis']


# In[16]:


# splitting data into training and test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)


# In[17]:


# scaling data

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# # Logistic Regression

# In[18]:


# fitting data to model

from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)


# In[19]:


# model predictions

y_pred = log_reg.predict(X_test)


# In[20]:


# accuracy score

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print(accuracy_score(y_train, log_reg.predict(X_train)))

log_reg_acc = accuracy_score(y_test, log_reg.predict(X_test))
print(log_reg_acc)


# In[21]:


# confusion matrix

print(confusion_matrix(y_test, y_pred))


# In[22]:


# classification report

print(classification_report(y_test, y_pred))


# # K Neighbors Classifier (KNN)

# In[23]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)


# In[24]:


# model predictions 

y_pred = knn.predict(X_test)


# In[25]:


# accuracy score

print(accuracy_score(y_train, knn.predict(X_train)))

knn_acc = accuracy_score(y_test, knn.predict(X_test))
print(knn_acc)


# In[26]:


# confusion matrix

print(confusion_matrix(y_test, y_pred))


# In[27]:


# classification report

print(classification_report(y_test, y_pred))


# # Support Vector Classifier (SVC)

# In[28]:


from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

svc = SVC()
parameters = {
    'gamma' : [0.0001, 0.001, 0.01, 0.1],
    'C' : [0.01, 0.05, 0.5, 0.1, 1, 10, 15, 20]
}

grid_search = GridSearchCV(svc, parameters)
grid_search.fit(X_train, y_train)


# In[29]:


# best parameters

grid_search.best_params_


# In[30]:


# best accuracy 

grid_search.best_score_


# In[31]:


svc = SVC(C = 10, gamma = 0.01)
svc.fit(X_train, y_train)


# In[32]:


# model predictions 

y_pred = svc.predict(X_test)


# In[33]:


# accuracy score

print(accuracy_score(y_train, svc.predict(X_train)))

svc_acc = accuracy_score(y_test, svc.predict(X_test))
print(svc_acc)


# In[34]:


# confusion matrix

print(confusion_matrix(y_test, y_pred))


# In[35]:


# classification report

print(classification_report(y_test, y_pred))


# # SGD Classifier

# In[36]:


from sklearn.linear_model import SGDClassifier

sgd = SGDClassifier()
parameters = {
    'alpha' : [0.0001, 0.001, 0.01, 0.1, 1],
    'loss' : ['hinge', 'log'],
    'penalty' : ['l1', 'l2']
}

grid_search = GridSearchCV(sgd, parameters, cv = 10, n_jobs = -1)
grid_search.fit(X_train, y_train)


# In[37]:


# best parameter 

grid_search.best_params_


# In[38]:


sgd = SGDClassifier(alpha = 0.001, loss = 'log', penalty = 'l2')
sgd.fit(X_train, y_train)


# In[39]:


# model predictions 

y_pred = sgd.predict(X_test)


# In[40]:


# accuracy score

print(accuracy_score(y_train, sgd.predict(X_train)))

sgd_acc = accuracy_score(y_test, sgd.predict(X_test))
print(sgd_acc)


# In[41]:


# confusion matrix

print(confusion_matrix(y_test, y_pred))


# In[42]:


# classification report

print(classification_report(y_test, y_pred))


# # Decision Tree Classifier

# In[43]:


from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()

parameters = {
    'criterion' : ['gini', 'entropy'],
    'max_depth' : range(2, 32, 1),
    'min_samples_leaf' : range(1, 10, 1),
    'min_samples_split' : range(2, 10, 1),
    'splitter' : ['best', 'random']
}

grid_search_dt = GridSearchCV(dtc, parameters, cv = 5, n_jobs = -1, verbose = 1)
grid_search_dt.fit(X_train, y_train)


# In[44]:


# best parameters

grid_search_dt.best_params_


# In[45]:


# best score

grid_search_dt.best_score_


# In[46]:


dtc = DecisionTreeClassifier(criterion = 'entropy', max_depth = 28, min_samples_leaf = 1, min_samples_split = 8, splitter = 'random')
dtc.fit(X_train, y_train)


# In[47]:


y_pred = dtc.predict(X_test)


# In[48]:


# accuracy score

print(accuracy_score(y_train, dtc.predict(X_train)))

dtc_acc = accuracy_score(y_test, dtc.predict(X_test))
print(dtc_acc)


# In[49]:


# confusion matrix

print(confusion_matrix(y_test, y_pred))


# In[50]:


# classification report

print(classification_report(y_test, y_pred))


# # Random Forest Classifier

# In[51]:


from sklearn.ensemble import RandomForestClassifier

rand_clf = RandomForestClassifier(criterion = 'entropy', max_depth = 11, max_features = 'auto', min_samples_leaf = 2, min_samples_split = 3, n_estimators = 130)
rand_clf.fit(X_train, y_train)


# In[52]:


y_pred = rand_clf.predict(X_test)


# In[53]:


# accuracy score

print(accuracy_score(y_train, rand_clf.predict(X_train)))

ran_clf_acc = accuracy_score(y_test, y_pred)
print(ran_clf_acc)


# In[54]:


# confusion matrix

print(confusion_matrix(y_test, y_pred))


# In[55]:


# classification report

print(classification_report(y_test, y_pred))


# # Voting Classifier

# In[56]:


from sklearn.ensemble import VotingClassifier

classifiers = [('Logistic Regression', log_reg), ('K Nearest Neighbours', knn), ('Support Vector Classifier', svc),
               ('Decision Tree', dtc)]

vc = VotingClassifier(estimators = classifiers)

vc.fit(X_train, y_train)


# In[57]:


y_pred = vc.predict(X_test)


# In[58]:


# accuracy score

print(accuracy_score(y_train, vc.predict(X_train)))

vc_acc = accuracy_score(y_test, y_pred)
print(vc_acc)


# In[59]:


# confusion matrix

print(confusion_matrix(y_test, y_pred))


# In[60]:


# classification report

print(classification_report(y_test, y_pred))


# # Ada Boost Classifier

# In[61]:


from sklearn.ensemble import AdaBoostClassifier

ada = AdaBoostClassifier(base_estimator = dtc)

ada = AdaBoostClassifier(dtc, n_estimators = 180)
ada.fit(X_train, y_train)


# In[62]:


y_pred = ada.predict(X_test)


# In[63]:


# accuracy score

print(accuracy_score(y_train, ada.predict(X_train)))

ada_acc = accuracy_score(y_test, y_pred)
print(ada_acc)


# In[64]:


# confusion matrix

print(confusion_matrix(y_test, y_pred))


# In[65]:


# classification report

print(classification_report(y_test, y_pred))


# # Gradient Boosting Classifier

# In[66]:


from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier()

parameters = {
    'loss': ['deviance', 'exponential'],
    'learning_rate': [0.001, 0.1, 1, 10],
    'n_estimators': [100, 150, 180, 200]
}

grid_search_gbc = GridSearchCV(gbc, parameters, cv = 5, n_jobs = -1, verbose = 1)
grid_search_gbc.fit(X_train, y_train)


# In[67]:


# best parameters 

grid_search_gbc.best_params_


# In[68]:


# best score

grid_search_gbc.best_score_


# In[69]:


gbc = GradientBoostingClassifier(learning_rate = 1, loss = 'exponential', n_estimators = 200)
gbc.fit(X_train, y_train)


# In[70]:


y_pred = gbc.predict(X_test)


# In[71]:


# accuracy score

print(accuracy_score(y_train, gbc.predict(X_train)))

gbc_acc = accuracy_score(y_test, y_pred)
print(gbc_acc)


# In[72]:


# confusion matrix

print(confusion_matrix(y_test, y_pred))


# In[73]:


# classification report

print(classification_report(y_test, y_pred))


# # Stochastic Gradient Boosting (SGB)

# In[74]:


sgbc = GradientBoostingClassifier(max_depth=4, subsample=0.9, max_features=0.75, n_estimators=200, random_state=0)

sgbc.fit(X_train, y_train)


# In[75]:


y_pred = sgbc.predict(X_test)


# In[76]:


# accuracy score

print(accuracy_score(y_train, sgbc.predict(X_train)))

sgbc_acc = accuracy_score(y_test, y_pred)
print(sgbc_acc)


# In[77]:


# confusion matrix

print(confusion_matrix(y_test, y_pred))


# In[78]:


# classification report

print(classification_report(y_test, y_pred))


# # Extreme Gradient Boosting

# In[79]:


from xgboost import XGBClassifier 

xgb = XGBClassifier(objective = 'binary:logistic', learning_rate = 0.5, max_depth = 5, n_estimators = 180)

xgb.fit(X_train, y_train)


# In[80]:


y_pred = xgb.predict(X_test)


# In[81]:


# accuracy score

print(accuracy_score(y_train, xgb.predict(X_train)))

xgb_acc = accuracy_score(y_test, y_pred)
print(xgb_acc)


# In[82]:


# confusion matrix

print(confusion_matrix(y_test, y_pred))


# In[83]:


# classification report

print(classification_report(y_test, y_pred))


# In[84]:


models = pd.DataFrame({
    'Model': ['Logistic Regression', 'KNN', 'SVC', 'SGD Classifier', 'Decision Tree Classifier', 'Random Forest Classifier', 'Voting Classifier', 'Ada Boost Classifier',
             'Gradient Boosting Classifier', 'Stochastic Gradient Boosting', 'XgBoost'],
    'Score': [log_reg_acc, knn_acc, svc_acc, sgd_acc, dtc_acc, ran_clf_acc, vc_acc, ada_acc, gbc_acc, sgbc_acc, xgb_acc]
})

models.sort_values(by = 'Score', ascending = False)


# ### Best model for diagnosing breast cancer is "Gradient Boosting Classifier" with an accuracy of 98.8%.

# 
