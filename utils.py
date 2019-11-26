# libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
import seaborn as sns
from sklearn.model_selection import KFold
from tensorflow.python.keras.activations import tanh
from sklearn.metrics import r2_score

def visualize(disease = None, category = None):
    '''
     the function visualizes age against global cortical thickness for a dataset
     of patients with a given disease (if given), otherwise visualizes it for all the patients
    :disease: must be the name of a disease in the selected category
    :category: must be one of the values "DX_01_Cat", "DX_01_Sub", "DX_01"
    '''
    behavioral = pd.read_csv('data/Behavioral/cleaned/HBNFinalSummaries.csv')
    mri = pd.read_csv('data/MRI/structuralMRI/GlobalCorticalThickness.csv')
    if disease == None:
        age = behavioral[['Age', 'EID']]
        age = age.rename(columns={'EID': 'ID'})
        mri = mri[['ID', 'GlobalCorticalThickness']]
        # join over common patient_id
        dataset = pd.merge(mri, age, on='ID', how='inner')
        dataset = dataset.drop('ID', axis = 1)
        # visualize
        sns.regplot(x = 'GlobalCorticalThickness', y = 'Age', data = dataset).set_title('All Patients')
    else:
        age = behavioral[['Age', 'EID', category]]
        age = age.rename(columns={'EID': 'ID'})
        mri = mri[['ID', 'GlobalCorticalThickness']]
        # join over common patient_id
        dataset = pd.merge(mri, age, on='ID', how='inner')
        # select observations with disease of interest
        dataset = dataset.loc[dataset[category] == disease]
        dataset = dataset[['GlobalCorticalThickness', 'Age']]
        # visualize
        sns.regplot(x = 'GlobalCorticalThickness', y = 'Age', data = dataset).set_title('Patients with %s' %disease)
  
 

def create_dataset_age(select_disease = None, select_category = None):
    '''
    only uses the global MRI score
    if select_disease = None the columns DX_01_Cat, DX_01_Sub, DX_01 will also be
    present in the dataset. Otherwise, only patients with the given disease
    will be present in the dataset
    '''
    behavioral = pd.read_csv('data/Behavioral/cleaned/HBNFinalSummaries.csv')
    mri = pd.read_csv('data/MRI/structuralMRI/GlobalCorticalThickness.csv')
    if select_disease == None:         
        age = behavioral[['EID', 'Age', 'DX_01_Cat', 'DX_01_Sub', 'DX_01']]
        age = age.rename(columns={'EID': 'ID'})
        mri = mri[['ID', 'GlobalCorticalThickness']]
        # join over common patient_id
        dataset = pd.merge(mri, age, on='ID', how='inner')
        return dataset
    else:
        age = behavioral[['EID', 'Age', select_category]]
        age = age.rename(columns={'EID': 'ID'})
        mri = mri[['ID', 'GlobalCorticalThickness']]
        # join over common patient_id
        dataset = pd.merge(mri, age, on='ID', how='inner')
        dataset = dataset.loc[dataset[select_category] == select_disease]
        dataset = dataset.drop([select_category], axis = 1)
        return dataset


# NOTE: the MRI features have only 4 columns with the same names across them:
# {'BrainSegVolNotVent', 'ID', 'ScanSite', 'eTIV'}. We take this into consideration
# when building the following function
        
def create_dataset_mri(select_disease = None, select_category = None, SCORE = 'Age', thickness= True, volume=True, subcortical=True, DTI = False):    
    '''
    from the behavioral data we select SCORE as a response (could be age, WISC, SWAN...)
    if select_disease = None the columns DX_01_Cat, DX_01_Sub, DX_01 will also be
    present in the dataset. Otherwise, only patients with the given disease
    will be present in the dataset
    '''
    behavioral = pd.read_csv('data/Behavioral/cleaned/HBNFinalSummaries.csv')
    
    dti = pd.read_csv('data/MRI/DTI/FAPerTract.csv')
    
    mri_global = pd.read_csv('data/MRI/structuralMRI/GlobalCorticalThickness.csv')
    mri_left = pd.read_csv('data/MRI/structuralMRI/CorticalThicknessLHROI.csv')
    mri_right = pd.read_csv('data/MRI/structuralMRI/CorticalThicknessRHROI.csv')
    mri_vol_left = pd.read_csv('data/MRI/structuralMRI/CorticalVolumeLHROI.csv')
    mri_vol_right = pd.read_csv('data/MRI/structuralMRI/CorticalVolumeRHROI.csv')
    mri_sub_left = pd.read_csv('data/MRI/structuralMRI/SubCorticalVolumeLHROI.csv')
    mri_sub_right = pd.read_csv('data/MRI/structuralMRI/SubCorticalVolumeRHROI.csv')
    if thickness == True and volume == True and subcortical == True:
        dfs = [mri_left, mri_right, mri_vol_left, mri_vol_right, mri_sub_left, mri_sub_right]
    elif thickness == True and volume == True and subcortical == False:
        dfs = [mri_left, mri_right, mri_vol_left, mri_vol_right]
    elif thickness == True and volume == False and subcortical == False:
        dfs = [mri_left, mri_right]
    elif thickness == False and volume == True and subcortical == False:
        dfs = [mri_vol_left, mri_vol_right]
    elif thickness == False and volume == False and subcortical == True:
        dfs = [mri_sub_left, mri_sub_right]
    elif thickness == False and volume == True and subcortical == True:
        dfs = [mri_vol_left, mri_vol_right, mri_sub_left, mri_sub_right]
    else:
        raise ValueError('This combination of features is not allowed.')
    merged = reduce(lambda left,right: pd.merge(left,right,on=['ID', 'ScanSite', 'eTIV', 'BrainSegVolNotVent']), dfs)
    merged = pd.merge(merged, mri_global, on=['ID', 'ScanSite'])
    MRI = merged.drop(['ScanSite'], axis = 1)

    if select_disease == None:
        score = behavioral[['EID', SCORE, 'DX_01_Cat', 'DX_01_Sub', 'DX_01']]
        score = score.rename(columns={'EID': 'ID'})
        # join over common patient_id
        dataset = pd.merge(score, MRI, on='ID', how='inner')
        if DTI == False:
            return dataset
        else:
            dataset = pd.merge(dataset, dti, on = 'ID', how = 'inner')
            dataset = dataset.drop('ScanSite', axis = 1)
            return dataset
    else:
        score = behavioral[['EID', SCORE, select_category]]
        score = score.rename(columns={'EID': 'ID'})
        # join over common patient_id
        dataset = pd.merge(score, MRI, on='ID', how='inner')
        dataset = dataset.loc[dataset[select_category] == select_disease]
        dataset = dataset.drop([select_category], axis = 1)
        if DTI == False:
            return dataset
        else:
            dataset = pd.merge(dataset, dti, on = 'ID', how = 'inner')
            dataset = dataset.drop('ScanSite', axis = 1)
            return dataset
        

def create_dataset_eeg(disease = None, category = None, SCORE = 'Age',
                       clusters = False, spectro=True, micro = False, PSD=False, ratios = False):    
    '''
    From the behavioral data we select SCORE as a response (could be age, WISC, SWAN...)
    If select_disease = None the columns DX_01_Cat, DX_01_Sub, DX_01 will also be
    present in the dataset. Otherwise, only patients with the given disease
    will be present in the dataset
    '''
    behavioral = pd.read_csv('data/Behavioral/cleaned/HBNFinalSummaries.csv')
    spectro_average = pd.read_csv('data/EEG/RestingEEG_Spectro_Average.csv')       
    spectro_clusters = pd.read_csv('data/EEG/RestingEEG_Spectro_Cluster.csv')
    microstates = pd.read_csv('data/EEG/RestingEEG_Microstates.csv')
    PSD_average = pd.read_csv('data/EEG/RestingEEG_PSD_Average.csv')
    PSD_clusters = pd.read_csv('data/EEG/RestingEEG_PSD_Cluster.csv')
    ratio = pd.read_csv('data/EEG/RestingEEG_Spectro_Ratios.csv')
    
    if clusters == False:
        if PSD == False:
            EEG = spectro_average
        else:
            EEG = reduce(lambda left,right: pd.merge(left,right,on='id'), [spectro_average, PSD_average])
    elif clusters == True:
        if PSD == False:
            EEG = spectro_clusters
        else:
            EEG = reduce(lambda left,right: pd.merge(left,right,on='id'), [spectro_clusters, PSD_clusters])
    if micro == True:
        EEG = reduce(lambda left,right: pd.merge(left,right,on='id'), [EEG, microstates])
    if ratios == True:
        EEG = reduce(lambda left,right: pd.merge(left,right,on='id'), [EEG, ratio])

    if disease == None:
        score = behavioral[['EID', SCORE, 'DX_01_Cat', 'DX_01_Sub', 'DX_01']]
        score = score.rename(columns={'EID': 'id'})
        # join over common patient_id
        dataset = pd.merge(score, EEG, on='id', how='inner')
        return dataset
    else:
        score = behavioral[['EID', SCORE, category]]
        score = score.rename(columns={'EID': 'id'})
        # join over common patient_id
        dataset = pd.merge(score, EEG, on='id', how='inner')
        dataset = dataset.loc[dataset[category] == disease]
        dataset = dataset.drop([category], axis = 1)
        return dataset
    
            
        
        
        
def create_dataset_eeg_old(disease = None, category = None, SCORE = 'Age', clusters = False, channels = False):    
    '''
    EEG data. Average eeg data is always loaded. Clusters and channels control if want to include
    these eeg features also.
    From the behavioral data we select SCORE as a response (could be age, WISC, SWAN...)
    If select_disease = None the columns DX_01_Cat, DX_01_Sub, DX_01 will also be
    present in the dataset. Otherwise, only patients with the given disease
    will be present in the dataset
    '''
    behavioral = pd.read_csv('data/Behavioral/cleaned/HBNFinalSummaries.csv')
    eeg_average = pd.read_csv("data/EEG/resting_eeg_average.csv")         # 62 features
    eeg_clusters = pd.read_csv("data/EEG/resting_eeg_clusters.csv")       # 302 features    
    eeg_channels = pd.read_csv("data/EEG/resting_eeg_channels.csv")       # 5054 features

    common_columns = ['eyesclosed_alphapeak_derivative_amplitude',
        'eyesclosed_alphapeak_derivative_freq',
        'eyesclosed_alphapeak_gravity_amplitude',
        'eyesclosed_alphapeak_gravity_freq',
        'eyesclosed_alphapeak_max_amplitude',
        'eyesclosed_alphapeak_max_freq',
        'eyesopen_alphapeak_derivative_amplitude',
        'eyesopen_alphapeak_derivative_freq',
        'eyesopen_alphapeak_gravity_amplitude',
        'eyesopen_alphapeak_gravity_freq',
        'eyesopen_alphapeak_max_amplitude',
        'eyesopen_alphapeak_max_freq',
        'id',
        'quality_rating']
    
    if clusters == False and channels == False:
        EEG = eeg_average
    elif clusters == True and channels == False:
        dfs = [eeg_average, eeg_clusters]
        EEG = reduce(lambda left,right: pd.merge(left,right,on=common_columns), dfs)
    elif clusters == True and channels == True:
        dfs = [eeg_average, eeg_clusters, eeg_channels]
        EEG = reduce(lambda left,right: pd.merge(left,right,on=common_columns), dfs)
    else:
        raise ValueError('If you load channels you should also load clusters')
    if disease == None:
        score = behavioral[['EID', SCORE, 'DX_01_Cat', 'DX_01_Sub', 'DX_01']]
        score = score.rename(columns={'EID': 'id'})
        # join over common patient_id
        dataset = pd.merge(score, EEG, on='id', how='inner')
        return dataset
    else:
        score = behavioral[['EID', SCORE, category]]
        score = score.rename(columns={'EID': 'id'})
        # join over common patient_id
        dataset = pd.merge(score, EEG, on='id', how='inner')
        dataset = dataset.loc[dataset[category] == disease]
        dataset = dataset.drop([category], axis = 1)
        return dataset
    
    
    
# Helper function for cross-validation
def cv(model, data, labels, n_splits = 5, want_r2=False):
    '''
    model: must be a sklearn object with .fit and .predict methods
    data: the X matrix containing the features, can be a pd.DataFrame or a np object (array or matrix)
    labels: y, can be a pd.DataFrame or a np array
    n_splits: number of desired folds
    => returns array of mean suqared error calculated on each fold
    '''
    kf = KFold(n_splits=n_splits)
    data = np.array(data)
    labels = np.array(labels)
    mses = []
    r2s = []
    i = 1
    for train, test in kf.split(data):
        print("Split: {}".format(i), end="\r")
        X_train, X_test, y_train, y_test = data[train], data[test], labels[train], labels[test]
        model.fit(X=X_train, y=y_train)
        pred = model.predict(X_test)
        mse = sum((pred - y_test)**2)/len(test)
        r2 = r2_score(y_pred=pred, y_true=y_test)
        mses.append(mse)
        r2s.append(r2)
        i = i+1
        
    if want_r2:
        return (mses, r2s)
    else:
        return mses
    
    
# example usages
if __name__ == '__main__':
    # visualizations
    visualize(disease = 'Attention-Deficit/Hyperactivity Disorder', category = 'DX_01_Sub')
    visualize()
    # dataset with age, major disorder labels (DX_01_Cat, DX_01_Sub, DX_01) and ONLY global mri score
    data1 = create_dataset_age()
    # dataset with age, ALL MRI high-level features and major disorder labels (DX_01_Cat, DX_01_Sub, DX_01) for ALL patients
    data2 = create_dataset_mri(SCORE = 'Age', thickness = True, volume = False, subcortical = False)
    # dataset with age and ALL MRI high-level features ONLY for patients with Attention-Deficit/Hyperactivity Disorder
    data3 = create_dataset_mri(select_disease = 'Attention-Deficit/Hyperactivity Disorder', select_category = 'DX_01_Sub', SCORE = 'Age')