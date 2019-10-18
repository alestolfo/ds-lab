# libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce

def visualize(disease = None, category = None):
    '''
     the function visualizes age against global cortical thickness for a dataset
     of patients with a given disease (if given), otherwise visualizes it for all the patients
    :disease: must be the name of a disease in the selected category
    :category: must be one of the values "DX_01_Cat", "DX_01_Sub", "DX_01"
    '''
    behavioral = pd.read_csv('data/Behavioral/AllData.csv')
    mri = pd.read_csv('data/structuralMRI/GlobalCorticalThickness.csv')
    if disease == None:
        age = behavioral[['Age', 'Patient_ID']]
        age = age.rename(columns={'Patient_ID': 'ID'})
        mri = mri[['ID', 'GlobalCorticalThickness']]
        # join over common patient_id
        dataset = pd.merge(mri, age, on='ID', how='inner')
        # define X and y    
        X = dataset[['GlobalCorticalThickness']]
        y = dataset[['Age']] 
        # convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        # visualize
        plt.scatter(X, y)
        plt.title('All patients')
        plt.xlabel('Cortical thickness')
        plt.ylabel('Age')
        plt.show()
    else:
        age = behavioral[['Age', 'Patient_ID', category]]
        age = age.rename(columns={'Patient_ID': 'ID'})
        mri = mri[['ID', 'GlobalCorticalThickness']]
        # join over common patient_id
        dataset = pd.merge(mri, age, on='ID', how='inner')
        # select observations with disease of interest
        dataset = dataset.loc[dataset[category] == disease]
        # define X and y    
        X = dataset[['GlobalCorticalThickness']]
        y = dataset[['Age']] 
        # convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        # visualize
        plt.scatter(X, y)
        plt.title('Patients with %s' %disease)
        plt.xlabel('Cortical thickness')
        plt.ylabel('Age')
        plt.show()


# NOTE: the MRI features have only 4 columns with the same names across them:
# {'BrainSegVolNotVent', 'ID', 'ScanSite', 'eTIV'}. We take this into consideration
# when building the following function

def create_dataset_age(select_disease = None, select_category = None):
    '''
    only uses the global MRI score
    if select_disease = None the columns DX_01_Cat, DX_01_Sub, DX_01 will also be
    present in the dataset. Otherwise, only patients with the given disease
    will be present in the dataset
    '''
    behavioral = pd.read_csv('data/Behavioral/AllData.csv')
    if select_disease == None:         
        mri = pd.read_csv('data/structuralMRI/GlobalCorticalThickness.csv')
        age = behavioral[['Age', 'Patient_ID', 'DX_01_Cat', 'DX_01_Sub', 'DX_01']]
        age = age.rename(columns={'Patient_ID': 'ID'})
        mri = mri[['ID', 'GlobalCorticalThickness']]
        # join over common patient_id
        dataset = pd.merge(mri, age, on='ID', how='inner')
        return dataset
    else:
        mri = pd.read_csv('data/structuralMRI/GlobalCorticalThickness.csv')
        age = behavioral[['Age', 'Patient_ID', select_category]]
        age = age.rename(columns={'Patient_ID': 'ID'})
        mri = mri[['ID', 'GlobalCorticalThickness']]
        # join over common patient_id
        dataset = pd.merge(mri, age, on='ID', how='inner')
        dataset = dataset.loc[dataset[select_category] == select_disease]
        dataset = dataset.drop([select_category], axis = 1)
        return dataset

        
def create_dataset(select_disease = None, select_category = None, SCORE = 'Age'):    
    '''
    ALL the MRI high-level features are used
    from the behavioral data we select SCORE as a response (could be age, WISC, SWAN...)
    if select_disease = None the columns DX_01_Cat, DX_01_Sub, DX_01 will also be
    present in the dataset. Otherwise, only patients with the given disease
    will be present in the dataset
    '''
    behavioral = pd.read_csv('data/Behavioral/AllData.csv')
    
    mri_global = pd.read_csv('data/structuralMRI/GlobalCorticalThickness.csv')
    mri_left = pd.read_csv('data/structuralMRI/CorticalThicknessLHROI.csv')
    mri_right = pd.read_csv('data/structuralMRI/CorticalThicknessRHROI.csv')
    mri_vol_left = pd.read_csv('data/structuralMRI/CorticalVolumeLHROI.csv')
    mri_vol_right = pd.read_csv('data/structuralMRI/CorticalVolumeRHROI.csv')
    mri_sub_left = pd.read_csv('data/structuralMRI/SubCorticalVolumeLHROI.csv')
    mri_sub_right = pd.read_csv('data/structuralMRI/SubCorticalVolumeRHROI.csv')
    dfs = [mri_left, mri_right, mri_vol_left, mri_vol_right, mri_sub_left, mri_sub_right]
    merged = reduce(lambda left,right: pd.merge(left,right,on=['ID', 'ScanSite', 'eTIV', 'BrainSegVolNotVent']), dfs)
    merged = pd.merge(merged, mri_global, on=['ID', 'ScanSite'])
    MRI = merged.drop(['ScanSite'], axis = 1)

    if select_disease == None:
        score = behavioral[[SCORE, 'Patient_ID', 'DX_01_Cat', 'DX_01_Sub', 'DX_01']]
        score = score.rename(columns={'Patient_ID': 'ID'})
        # join over common patient_id
        dataset = pd.merge(score, MRI, on='ID', how='inner')
        return dataset
    else:
        score = behavioral[[SCORE, 'Patient_ID', select_category]]
        score = score.rename(columns={'Patient_ID': 'ID'})
        # join over common patient_id
        dataset = pd.merge(score, MRI, on='ID', how='inner')
        dataset = dataset.loc[dataset[select_category] == select_disease]
        dataset = dataset.drop([select_category], axis = 1)
        return dataset   
    
    
# example usages
if __name__ == '__main__':
    # visualizations
    visualize(disease = 'Attention-Deficit/Hyperactivity Disorder', category = 'DX_01_Sub')
    visualize()
    # dataset with age, major disorder labels (DX_01_Cat, DX_01_Sub, DX_01) and ONLY global mri score
    data1 = create_dataset_age()
    # dataset with age, ALL MRI high-level features and major disorder labels (DX_01_Cat, DX_01_Sub, DX_01) for ALL patients
    data2 = create_dataset(SCORE = 'Age')
    # dataset with age and ALL MRI high-level features ONLY for patients with Attention-Deficit/Hyperactivity Disorder
    data2 = create_dataset(select_disease = 'Attention-Deficit/Hyperactivity Disorder', select_category = 'DX_01_Sub', SCORE = 'Age')