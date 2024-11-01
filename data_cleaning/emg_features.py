# python code to extract EMG features

import numpy as np

"""
Written by Momona Yamagami, PhD 
Last updated: 2/15/2024

EMG features originally defined in: 
Abbaspour S, Lindén M, Gholamhosseini H, Naber A, Ortiz-Catalan M. Evaluation of surface EMG-based recognition algorithms for decoding hand movements. 
    Medical & biological engineering & computing. 2020 Jan;58:83-100.
    MAV, STD, DAMV, IAV, Var, WL, Cor, HMob, and HCom
IMU: only mean value extraction (MV)
    https://link.springer.com/article/10.1186/s12984-017-0284-4
    https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8932579

"""

"""
0) mean value (MV)
    MV = 1/N (\sum^N_{i=1}x_i)
    N: length of signal
    x_n: EMG signal in a segment
"""
def MV(s,fs=None):
    N = len(s)
    return 1/N*sum(s)

"""
1) mean absolute value (MAV)
    MAV = 1/N (\sum^N_{i=1}|x_i|)
    N: length of signal
    x_n: EMG signal in a segment
"""
def MAV(s,fs=None):
    N = len(s)
    return 1/N*sum(abs(s))

"""
2) standard deviation (STD)
    STD = \sqrt{1/(N-1) \sum^N_{i=1}(x_i-xbar)^2}
"""
def STD(s,fs=None):
    N = len(s)
    sbar = np.mean(s)
    return np.sqrt(1/(N-1)*sum((s-sbar)**2))

"""
3) variance of EMG (Var)
    Var = 1/(N-1)\sum^N_{i=1} x_i^2
"""
def Var(s,fs=None):
    N = len(s)
    return 1/(N-1)*sum(s**2)

"""
4) waveform length
    WL = sum (|x_i-x_{i-1})
"""
def WL(s,fs=None):
    return (sum(abs(s[1:]-s[:-1]))) / 1.0 # make sure convert to float64


"""
10) correlation coefficient (Cor)
    Cor(x,y)
    x, y: each pair of EMG channels in a time window
"""
def Cor(x,y,fs=None):
    xbar = np.mean(x)
    ybar = np.mean(y)
    num = abs(sum((x-xbar)*(y-ybar)))
    den = np.sqrt(sum((x-xbar)**2)*sum((y-ybar)**2))
    return num/den

""" 
11) Difference absolute mean value (DAMV)
    DAMV = 1/N sum_{i=1}^{N-1} |x_{i+1}-x_i|
"""
def DAMV(s,fs=None):
    N = len(s)
    return 1/N * sum(abs(s[1:]-s[:-1]))


"""
16) integrated absolute value (IAV)
    IAV = \sum^N_{i=1} |x_i|
"""
def IAV(s,fs=None):
    return (sum(abs(s))) / 1.0 # make sure convert to float64

"""
17) Hjorth mobility parameter (HMob)
    derivative correct?
"""
def HMob(s,fs=None):
    dt = 1/fs # 1/2000
    ds = np.gradient(s,dt) # compute derivative 
    return np.sqrt(Var(ds)/Var(s))

"""
18) Hjorth complexity parameter
    compares the similarity of the shape of a signal with
    pure sine wave 
    HCom = mobility(dx(t)/dt) / mobility(x(t))
"""
def HCom(s,fs=None):
    dt = 1/fs
    ds = np.gradient(s,dt)
    return HMob(ds,fs) / HMob(s,fs)
