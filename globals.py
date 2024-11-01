# Momona Yamagami

global_path = " " # home anonymized
temp_path = " " # anonymized
models_path = " " # anonymized

extract_path = "segmented_raw_data\\"
to_save_path = "segmented_filtered_data\\"
fmts = ['png']

# things to import
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import glob
import copy
import pandas as pd
from scipy.fft import fft, fftfreq
from scipy import signal
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from sklearn.decomposition import PCA,NMF
import seaborn as sns
from mdollar import Mdollar
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import wilcoxon,ttest_rel
import numpy.matlib
import seaborn as sns
from scipy.signal import periodogram, welch
from scipy.stats import zscore, mannwhitneyu
from data_cleaning.emg_features import *
from sklearn import svm, preprocessing
import torch
from torch.utils.data import Dataset, DataLoader,Subset
from torchvision import transforms, utils
from torch import nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import pickle as pickle
import time
from copy import deepcopy
import gc 
from itertools import combinations
import math as math

# plt.style.use('dark_background')
# for experimenter-defined

try:
    with open('data/participants.pickle', 'rb') as handle:
        _, pIDs_personalized,pIDs_standardized,pIDs_analogous,pIDs_all = pickle.load(handle)
except:
    print("Was not able to open the participant IDs")
    pass;

# import random
# N_seed = 42
# random.seed(N_seed)

# expert features 
corr_features_EMG = [Cor]
time_features_EMG = [MAV,STD,DAMV,IAV,Var,WL,HMob,HCom]
time_features_IMU = [MV]

motions = ['move','select-single','rotate','delete',
           'pan','close','zoom-in','zoom-out','open','duplicate',
           'gesture-1','gesture-2','gesture-3','gesture-4','gesture-5']
        #    ,'shrink','enlarge']

standard_motions = ['palm-pinch','two-handed-tap','point-and-pinch',
                    'air-tap','pinch-and-scroll']

n_pcs = 64 # num pcs
