# Create dataset of file paths, label and categories
# --------------------------------------------------

# pip install opencv-python

import pandas as pd
import numpy as np
import os

labels = ['NORMAL', 'PNEUMONIA']

def get_data(dir_path, csv_path, category):
    '''
    Creates dataset (csv file) mapping the file paths to
        their label (Normal or Pneumonia)
    
    Input: 
        dir_path (str): directory path of chest x-ray images
        csv_path (str): directory where to save csv file
        category (str): train, test or val
    
    Returns: Numpy array of dataset (and saves data as csv file)
    '''

    assert category in ['train', 'val', 'test'], \
        'Category can be either train, val or test'

    data = []
    for i, label in enumerate(labels):
        path = os.path.join(dir_path, label)
        label_num = i   #PNEUMONIA = 1, NORMAL = 0
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            data.append([file_path, label_num])
    
    dataset = np.array(data) 

    pd.DataFrame(dataset).to_csv(csv_path + "data_" + category + ".csv", index=False)

    return dataset







