# libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score

# load the data
behavioral = pd.read_csv("data/Behavioral/AllData.csv")
# eeg_average = pd.read_csv("data/EEG/resting_eeg_average.csv")
# eeg_channels = pd.read_csv("data/EEG/resting_eeg_channels.csv")
# eeg_clusters = pd.read_csv("data/EEG/resting_eeg_clusters.csv")

# get all the columns names
# list(behavioral.columns.values)

# general info
shape = behavioral.shape  #2096, 7042
behavioral.count()
behavioral['DX_01_Cat'].nunique()  # 16 different categories in DX_01_Cat

# remove all first columns and labels, except from sex, age, study-site and major disturb cathegory
first_cols = behavioral.iloc[:, :170]
behavioral = behavioral.iloc[:, 170:]
# list(first_cols.columns.values)
behavioral = behavioral.join(first_cols[['Sex', 'Age', 'Study.Site', 'DX_01_Cat']])

# remove observations with no label for DX_01_Cat
behavioral = behavioral.dropna(subset=['DX_01_Cat'])

# separate train and test set
# shuffle the rows
behavioral = behavioral.sample(frac=1)

# separate labels from features
labels = behavioral[['DX_01_Cat']]
behavioral = behavioral.drop(['DX_01_Cat'], axis=1)
# at this point behavioral.shape = (1402, 6875)

# remove cathegorical variables
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
behavioral = behavioral.select_dtypes(include=numerics)          # behavioral.shape = (1402, 5731)

# check total number of NA values
behavioral.isna().sum().sum()

# convert to numpy arrays
behavioral = np.array(behavioral)
labels = np.array(labels)

# replace NA with mean
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
behavioral = imp.fit_transform(behavioral)  
# verify no more NA
np.isnan(behavioral).sum()       # 0

# keep about 30% for test set
n = np.int(behavioral.shape[0]*0.33)
behavioral_test = behavioral[:n,:]
behavioral = behavioral[n:,:]
labels = labels[n:,:]
labels_test = labels[:n,:]
labels = np.reshape(labels, [labels.shape[0],])
labels_test = np.reshape(labels_test, [labels_test.shape[0],])

# try random-forest classifier
clf = RandomForestClassifier(n_estimators=2000, max_depth = 1000, criterion='gini',
                             min_samples_split= 0.65, min_samples_leaf=0.15,
                             max_features = 500, oob_score = True, n_jobs = -1,
                             random_state =50, verbose = 1)
clf = clf.fit(behavioral, labels)
predictions = clf.predict(behavioral_test)

# estimate test error with CV
cv_score = cross_val_score(clf, behavioral, labels, cv=10)
estimated_error = np.mean(cv_score)
estimated_error

# compute accuracy on our test split
vect = (predictions == labels_test)*1
accuracy = sum(vect) / predictions.shape[0]
accuracy
