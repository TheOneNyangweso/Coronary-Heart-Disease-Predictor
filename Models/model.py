# %% [markdown]
# Standard imports

# %%
# The Libraries needed for perfoming analysis and model development and evaluation
# classifcation_report calculates precision_score, recall_score and f1_score with just one line,
# making the individual libraries not necessary

%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import roc_auc_score
import missingno as msno

# %%
# The absolute path to the file where dataset is located

path = '/home/nyangweso/Desktop/Ds_1/Coronary-Heart-Disease-Predictor/Data/framingham.csv'

# %% [markdown]
# ### Data Preprocessing, feature engineering and feature selection

# %%
# Loading data and displaying the first 10 records

df = pd.read_csv(path)
df.head(10)

# %% [markdown]
# Getting shape of dataframe

# %%
df.shape

# %% [markdown]
# Getting a concise summary of the DataFrame.

# %%
df.info()

# %% [markdown]
# Using msno.bar(df) to get a simple visualization of nullity by column:

# %%
msno.bar(df)

# %% [markdown]
# Getting the nullity in each column by percentage

# %%
data = pd.DataFrame((df.isna().mean() * 100).round(2), columns=['count_%'])
data

# %% [markdown]
# We can see that the columns with missing values are education, cigsPerDay, BPMeds, totChol, BMI, heartRate and glucose.<br> 
# The columns will be cleaned differently

# %% [markdown]
# First lets look to see if they is a correlation between currentSmoker column and cigsPerDay

# %%
df.corr()

# %%
df.loc[(df['currentSmoker']) & (df['cigsPerDay'].isna())]

# %% [markdown]
# We can see that all null values found in cigsPerDay columns are located where currentSmoker is 1.<br> 
# This will helpful in knowing what method to use when cleaning data in cigsPerDay column

# %%
df.columns

# %% [markdown]
# ### Method 1 : Filling missing values with zero 
# The null values in the education values might indicate that the person does not have any education background<br> 
# Thus we can simply fill all null values with zero 

# %%
df['education'] = df['education'].fillna(0)

# %% [markdown]
# ### Method 2 : Dropping records in columns
# For columns whose percantage of missing values is less than 2% and have low correlations with other columns, we can simply drop the records since the impact on the datasey size will be small

# %%
columns_to_drop = ['BPMeds', 'totChol', 'BMI', 'heartRate']

# %%
len(df[columns_to_drop].dropna()) / len(df)

# %% [markdown]
# We can confirm that less than 3% of the sample size will be lost. This is acceptable

# %%
df.dropna(subset=columns_to_drop, inplace=True)

# %% [markdown]
# ### Method 3 : Imputing missing values with mean
# For the remaining columns, we will fill the missing values with the means of the respective columns

# %%
mean_of_cigsPerDay = df.loc[(
    df['cigsPerDay'] > 0)]['cigsPerDay'].mean().round()
mean_of_cigsPerDay

# %%
df['cigsPerDay'] = df['cigsPerDay'].fillna(mean_of_cigsPerDay)

# %%
fig = plt.figure()
ax = fig.add_subplot(111)

# original data
pd.read_csv(path)['cigsPerDay'].hist(bins=50, ax=ax, density=True, color='red')
df['cigsPerDay'].hist(
    bins=50, ax=ax, density=True, color='blue', alpha=0.8)

# %%
df['glucose'] = df['glucose'].fillna(
    df['glucose'].mean(numeric_only=True).round())

# %%
fig = plt.figure()
ax = fig.add_subplot(111)

pd.read_csv(path)['glucose'].hist(bins=50, ax=ax, density=True, color='red')
df['glucose'].hist(
    bins=50, ax=ax, density=True, color='blue', alpha=0.8)

# %%
fig = plt.figure()
ax = fig.add_subplot(111)

pd.read_csv(path)['glucose'].plot(kind='kde', ax=ax, color='red')
df['glucose'].plot(kind='kde', ax=ax, color='green')

# %% [markdown]
# ### Feature Engineering and Selection

# %%
df.corr()

# %%
sns.heatmap(df.corr())

# %% [markdown]
# By the looks of both the correlation table and heatmap, the education column has no impact on the output column (TenYearCHD)<br>
# We can simply drop it

# %%
df.drop(["education"], axis=1, inplace=True)

# %%
df.shape

# %% [markdown]
# Let's now take a look at the count of unique values in the output column

# %%
df['TenYearCHD'].value_counts()

# %% [markdown]
# Splitting the dataframe into feature matrix X and output variable y

# %%
X = df.drop(columns=['TenYearCHD'])
y = df['TenYearCHD']

# %% [markdown]
# The imbalanced distribution of classes (613 instances of class 1 and 3506 instances of class 0) in the TenYearCHD column can pose challenges for machine learning models. \
# When dealing with imbalanced datasets, there are several strategies one can consider. For this case we chose to resample the data.\
# In particular, oversampling (Increasing the number of instances from the under-represented class by duplicating or generating synthetic examples) was the preffered choice \
# Oversampling has a number of advantages such as :
# - It retains information
# - It can help reduce bias towards the majority class

# %%
oversample = RandomOverSampler(sampling_strategy='minority')
X, y = oversample.fit_resample(X, y)

# %%
y.value_counts()

# %% [markdown]
# Splitting data into train and test sets

# %%
xtrain, xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.35, random_state=42)
xtrain.shape, xtest.shape, ytrain.shape, ytest.shape

# %%
ytrain = pd.DataFrame(ytrain)
ytest = pd.DataFrame(ytest)

# %% [markdown]
# Finally, we will standardize the features of the dataset using StandardScaler()

# %%
standard = StandardScaler()
xtrain = standard.fit_transform(xtrain)
xtest = standard.fit_transform(xtest)

# %% [markdown]
# ### Model Development

# %%
rf = RandomForestClassifier(n_estimators=80, max_depth=30, random_state=42)
rf.fit(xtrain, ytrain.values.ravel())
ypred_rf = rf.predict(xtest)
rf_model = accuracy_score(ytest, ypred_rf)
ypred_rf2 = rf.predict(xtrain)

# %% [markdown]
# ### Model Evaluation

# %% [markdown]
# 1. Accuracy

# %%
rf_model2 = accuracy_score(ytrain, ypred_rf2)
print(
    f"accuracy for test set :{rf_model:.2f}\naccuracy for train set :{rf_model2:.2f}")

# %% [markdown]
# 2. precision, recall, f1-score

# %%
print(classification_report(ytest, ypred_rf))

# %%
X.columns

# %%
feature_importances = rf.feature_importances_
feature_importances

# %% [markdown]
# 3. Confusion matrix

# %%
cm = confusion_matrix(ytest, ypred_rf)


plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=['Predicted Negative', 'Predicted Positive'],
            yticklabels=['Actual Negative', 'Actual Positive'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# %% [markdown]
# 4. ROC-AUC score

# %%
roc_auc = roc_auc_score(ytest, ypred_rf)
print(f'ROC-AUC Score: {roc_auc:.4f}')

# %% [markdown]
# The model perfoms well on various perfomance metrics hence no need to tune our model further.

# %% [markdown]
# ### Hyperparameter tuning using GridSearchCV()
# This is just a snippet of how we would have used GridSearchCV() to tune our model

# %%
param_grid = {'n_estimators': [a for a in range(0, 200, 20)], 'max_depth': [
    None, 10, 20, 30]}

grid_search = GridSearchCV(RandomForestClassifier(
    random_state=42), param_grid, cv=5)
grid_search.fit(xtrain, ytrain.values.ravel())

best_params = grid_search.best_params_
best_params

# %% [markdown]
# ### Saving our model

# %%
import pickle

# Saving the model to a file
with open('/home/nyangweso/Desktop/Ds_1/Coronary-Heart-Disease-Predictor/Models/CHD.pkl', 'wb') as model_file:
    pickle.dump(rf, model_file)

# To load the model from the file
# with open('/home/nyangweso/Desktop/Ds_1/Coronary-Heart-Disease-Predictor/Models/CHD.pkl', 'rb') as model_file:
#     loaded_model = pickle.load(model_file)


