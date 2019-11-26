# libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
import seaborn as sns
from sklearn.model_selection import KFold
from cde.density_estimator import MixtureDensityNetwork
from tensorflow.python.keras.activations import tanh

    
def cv_for_cde(data, labels, name, std, n_splits = 5):
    '''
    model: must be a sklearn object with .fit and .predict methods
    data: the X matrix containing the features, can be a pd.DataFrame or a np object (array or matrix)
    labels: y, can be a pd.DataFrame or a np array
    n_splits: number of desired folds
    => returns array of mean suqared error calculated on each fold
    '''
    input_dim = data.shape[1]
    kf = KFold(n_splits=n_splits, shuffle=True)
    data = np.array(data)
    labels = np.array(labels)
    mses = []
    i = 1
    for train, test in kf.split(data):
        model = MixtureDensityNetwork(name=name + str(i),
                              ndim_x=input_dim,
                              ndim_y=1,
                              n_centers=10,
                              hidden_sizes=(16, 16),
                              hidden_nonlinearity=tanh,
                              n_training_epochs=1000,
                              x_noise_std=std,
                              y_noise_std=std
                             )
        
        print("Split: {}".format(i), end="\r")
        X_train, X_test, y_train, y_test = data[train], data[test], labels[train], labels[test]
        model.fit(X=X_train, Y=y_train, verbose=True)
        pred = model.mean_(X_test)
        pred = pred.reshape((-1,1)).flatten()
        mse = sum((pred - y_test)**2)/len(test)
        print('MSE: {}'.format(mse))
        mses.append(mse)
        i = i+1
    return mses    
    
  