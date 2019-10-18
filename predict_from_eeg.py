# libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV


# data description:
# 3 level of detaildness: average (average measure over all electrodes),
# clusters (all electrodes clustered into 6 spatially distinct regions),
# channel (each measure for each of the 105 channel individually)

# load the data
eeg_average = pd.read_csv("data/EEG/resting_eeg_average.csv")
eeg_channels = pd.read_csv("data/EEG/resting_eeg_channels.csv")
eeg_clusters = pd.read_csv("data/EEG/resting_eeg_clusters.csv")
behavioral = pd.read_csv("data/Behavioral/AllData.csv")
# keep only id and label of interest
behavioral  = behavioral[['Patient_ID', 'DX_01_Cat']]
behavioral = behavioral.rename(columns={"Patient_ID": "id"})

# check total number of NA values
eeg_average.isna().sum().sum()       # 1662
eeg_channels.isna().sum().sum()      # 353978
eeg_clusters.isna().sum().sum()      # 1708

# stack all the dataframes in a single one
eeg = pd.concat([eeg_average, eeg_channels, eeg_clusters], axis = 1)

# remove duplicated columns (id, rating quality...)
eeg = eeg.loc[:,~eeg.columns.duplicated()]

# select id and corresponding labels from behavioral data
ids = eeg[['id']]
merged = ids.join(behavioral.set_index('id'), on='id')     # contains ids and labels

# add labels to eeg data
eeg = eeg.join(merged.set_index('id'), on = 'id')

# remove rows with na label
eeg = eeg.dropna(subset=['DX_01_Cat'])

# keep out factor columns
factors = eeg.select_dtypes(include='object')
eeg = eeg.select_dtypes(exclude='object')
quality = factors[['quality_rating']]
labels = factors[['DX_01_Cat']]

# convert to numpy array
eeg = np.array(eeg)
quality = np.array(quality)
labels = np.array(labels)
quality = np.reshape(quality, [quality.shape[0], 1])

# substitute missing values with the mean
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
eeg = imp.fit_transform(eeg)

# add quality again after hot-encoding it
# (IMPORTANT: the "drop = false" removes one column of dummy variables to avoid collinearity issues)
onehotencoder = OneHotEncoder(drop='first')
quality = onehotencoder.fit_transform(quality)
quality = quality.toarray()
eeg = np.concatenate((eeg, quality), axis = 1)

# keep about 30% for test set
X_train, X_test, y_train, y_test = train_test_split(eeg, labels, test_size=0.33, random_state=42)

# random-forest classifier
clf = RandomForestClassifier(n_estimators=800, max_depth = 80, criterion='gini',
                             min_samples_split= 0.25, min_samples_leaf=0.05,
                             max_features = 50, oob_score = True, n_jobs = -1,
                             random_state =5, verbose = 1, class_weight = 'balanced')


clf = clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

# estimate test error with CV
cv_score = cross_val_score(clf, X_train, y_train, cv=10)
estimated_error = np.mean(cv_score)
estimated_error                       

# compute accuracy on the test split
y_test = np.reshape(y_test, [y_test.shape[0],])
vect = (predictions == y_test)*1
accuracy = sum(vect) / predictions.shape[0]
print("accuracy: ", accuracy)       # 0.6692307692307692     <--- always the same because class imbalance, predict always the most frequent class
