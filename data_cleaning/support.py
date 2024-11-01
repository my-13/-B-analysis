from globals import * 

RESAMPLE_SIZE = 64
TOTAL_N_FILES = 300# 100 for experimenter-defined, 50 for user-defined, 50 for standardized, 41 for rehabilitation

def rectify_EMG(data):
    
    # 1. high-pass filter @ 40 hz, 4th order butterworth
    sos_high = signal.butter(N=2,Wn=40,btype='high',fs=2000,output='sos')
    hp_filtered = signal.sosfiltfilt(sos_high,data)
    
    # 2. demean and rectify
    emg_mean = hp_filtered - np.mean(hp_filtered)
    rectified = abs(emg_mean)
    
    # 3. low-pass filter @ 40 Hz, 4th order butterworth
    sos_low = signal.butter(N=2,Wn=40,btype='low',fs=2000,output='sos')
    lp_filtered = signal.sosfiltfilt(sos_low,rectified)
    return lp_filtered

def compute_SNR(s,fs=2000):
    nperseg = len(s)//4.5
    f,Pxx = welch(s,fs=2000,window='hamming',
        nperseg=nperseg,noverlap=nperseg//2,
        scaling='density',detrend=False)
    idx400 = list(abs(f-800)).index(min(abs(f-800)))
    N = len(f)-idx400
    noise_power = sum(Pxx[idx400:])/N*len(f)
    total_power = sum(Pxx)
    return 10*np.log10(total_power / noise_power)

def bandpass_EMG(data):
    sos_high = signal.butter(N=2,Wn=[10,500],btype='bandpass',fs=2000,output='sos')
    EMG_filt = signal.sosfiltfilt(sos_high,data)
    EMG_filt = EMG_filt - np.mean(EMG_filt)
    return EMG_filt

def movavg_EMG(data,window=100,overlap=0.5,fs=2000):
    # 200 ms window, overlap = 0 is no overlap, overlap=1 is complete overlap 
    N = int(1*window/1000*fs) # number of datapoints in one window
    N_overlap = int(N*overlap)
    
    movavg_data = []
    
    ix = N
    while ix < len(data):
        movavg_data.append(np.mean(data[ix-N:ix]))
        ix = ix + (N - N_overlap)
    
    return np.asarray(movavg_data)

def interaction_type(file):
    interaction = file.split("_")[4:6]
    return interaction

def resample(data,N):
    N_init = len(data)
    x = np.linspace(0,N_init-1,N)# x coordinates at which to evaluate interpolated values
    xp = np.linspace(0,N_init-1,N_init)# x coordinates of data points
    return np.interp(x,xp,data)

def get_info(filename,ix):
    if ix != 5:
        return filename.split("_")[ix]
    else:
        return filename.split("_")[ix].split(".")[0]

def read_and_save_cleaned_data(participantID,extract_path):#,df_EMG_exptr_def,df_EMG_usr_def,df_EMG_calib,df_EMG_rehab):
    path = temp_path[:-24]+ extract_path + participantID+"\\"
    save_path = temp_path + participantID + "\\"

    # make folders
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    if not os.path.isdir(save_path+'movavg_files'):
        os.mkdir(save_path+'movavg_files')
    if not os.path.isdir(save_path+'IMU_extract'):
        os.mkdir(save_path+'IMU_extract')
    if not os.path.isdir(save_path+'filtered_files'):
        os.mkdir(save_path+'filtered_files')

    data = pd.read_csv(temp_path[:-24] + "fnames/"+participantID+"_fnames.csv")
    EMG_files = data['fname_EMG'].apply(lambda x: x[:-3]+"csv").values
    IMU_files = data['fname_IMU'].apply(lambda x: x[:-3]+"csv").values
    
    for ix,file in enumerate(EMG_files):
        
        df_EMG = pd.read_csv(path+file,header=0,index_col=0)

        # filter raw EMG signals
        df_EMG_filt = df_EMG.apply(rectify_EMG,axis=0)

        # save as csv files
        df_EMG_filt.to_csv(save_path+'filtered_files/' + file[:-4] + "_filt.csv")

        # UNCOMMENT REST IF NEEDED
    #     # compute moving average time x 16 channels
    #     df_EMG_movavg = df_EMG_filt.apply(movavg_EMG,axis=0)
        
    #     # save as csv files
    #     df_EMG_movavg.to_csv(save_path+'movavg_files/' + file[:-4] + "_movavg.csv")


    # # analyze IMU data 
        
    # for ix,file in enumerate(IMU_files):
        
    #     df_IMU = pd.read_csv(path+file,header=0,index_col=0)
        
    #     # save just IMU data
    #     df_IMU.to_csv(save_path+'IMU_extract/' + file[:-4] + "_IMU_extract.csv")

def extract_vanillaNN_features(participantID):#,df_EMG_exptr_def,df_EMG_usr_def,df_EMG_calib,df_EMG_rehab):
    path = temp_path + participantID+"\\"
    save_path = temp_path + participantID + "\\"

    # make folders
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    if not os.path.isdir(save_path+'vanillaNN_files'):
        os.mkdir(save_path+'vanillaNN_files')

    data = pd.read_csv(temp_path[:-24] + "fnames/"+participantID+"_fnames.csv")
    EMG_files = data['fname_EMG'].apply(lambda x: x[:-4]+"_movavg.csv").values
    IMU_files = data['fname_IMU'].apply(lambda x: x[:-4]+"_IMU_extract.csv").values
    
    for ix,file in enumerate(EMG_files):
        
        # read in files
        df_EMG = pd.read_csv(path+"movavg_files/"+file,header=0,index_col=0)
        df_IMU = pd.read_csv(path+"IMU_extract/"+IMU_files[ix],header=0,index_col=0)

        # resample 
        RESAMPLE_SIZE = 64
        for ix,df in enumerate([df_EMG,df_IMU]):
            N_init = len(df)
            x = np.linspace(0,N_init-1,RESAMPLE_SIZE)# x coordinates at which to evaluate interpolated values
            xp = np.linspace(0,N_init-1,N_init)# x coordinates of data points
            df_temp = df.apply(lambda points: np.interp(x,xp,points))
            if ix==0:
                df_all = df_temp
            else:
                df_all = pd.concat([df_all,df_temp],ignore_index=True,axis=1)
        df_all = pd.melt(df_all,id_vars=None,value_vars=df_all.keys(),ignore_index=True)
        df_all= df_all.drop("variable",axis=1)
        df_all.to_csv(save_path+'vanillaNN_files/' + file[:-10] + "vanillaNN.csv",
                      index=False,header=False)
        
def extract_dollar_features(participantID):
    path = temp_path + participantID+"\\"
    save_path = temp_path + participantID + "\\"

    # make folders
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    if not os.path.isdir(save_path+'dollar_files'):
        os.mkdir(save_path+'dollar_files')

    data = pd.read_csv(temp_path[:-24] + "fnames/"+participantID+"_fnames.csv")
    EMG_files = data['fname_EMG'].apply(lambda x: x[:-4]+"_movavg.csv").values
    IMU_files = data['fname_IMU'].apply(lambda x: x[:-4]+"_IMU_extract.csv").values
    
    for ix,file in enumerate(EMG_files):
        
        # read in files
        df_EMG = pd.read_csv(path+"movavg_files/"+file,header=0,index_col=0)
        df_IMU = pd.read_csv(path+"IMU_extract/"+IMU_files[ix],header=0,index_col=0)

        # resample 
        RESAMPLE_SIZE = 64
        for ix,df in enumerate([df_EMG,df_IMU]):
            N_init = len(df)
            x = np.linspace(0,N_init-1,RESAMPLE_SIZE)# x coordinates at which to evaluate interpolated values
            xp = np.linspace(0,N_init-1,N_init)# x coordinates of data points
            df_temp = df.apply(lambda points: np.interp(x,xp,points))
            df_temp = df_temp.apply(lambda x: x - np.mean(x))
            stdev = df_temp.values.std()
            df_temp = df_temp.apply(lambda x: x / stdev)
            if ix==0:
                df_all = df_temp
            else:
                df_all = pd.concat([df_all,df_temp],ignore_index=True,axis=1)
        # df_all = pd.melt(df_all,id_vars=None,value_vars=df_all.keys(),ignore_index=True)
        # df_all= df_all.drop("variable",axis=1)
        df_all.to_csv(save_path+'dollar_files/' + file[:-10] + "dollar.csv",
                      index=False,header=False)

def extract_expert_features(participantID):
    path = temp_path[:-24]+ extract_path + participantID+"\\"
    save_path = temp_path + participantID + "\\"

    # make folders
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    if not os.path.isdir(save_path+'expert_files'):
        os.mkdir(save_path+'expert_files')

    data = pd.read_csv(temp_path[:-24] + "fnames/"+participantID+"_fnames.csv")
    EMG_files = data['fname_EMG'].apply(lambda x: x[:-3]+"csv").values
    IMU_files = data['fname_IMU'].apply(lambda x: x[:-3]+"csv").values
    
    for ix,file in enumerate(EMG_files):
        
        df_EMG = pd.read_csv(path+file,header=0,index_col=0)

        # filter raw EMG signals
        df_EMG_filt = df_EMG.apply(bandpass_EMG,axis=0)

        # time domain expert features
        df_features = pd.DataFrame()#(columns=df_EMG_filt.columns)
        for ix,time_feature in enumerate(time_features_EMG):
            temp_feature = pd.DataFrame(list(df_EMG_filt.apply(lambda s: time_feature(s.values,fs=2000),axis=0).values))
            df_features = df_features.append(temp_feature,ignore_index=True)
        # 128 features

        # COR EMG feature
        for ix,corr_feature in enumerate(corr_features_EMG):
            # NEED TO COMPUTE CORRELATION ACROSS ALL CHANNELS 
            chs = list(df_EMG_filt.keys()) # all EMG channel keys
            combs = list(combinations(chs,2)) # generate all possible combinations of channels 120 options
            temp_feature = []
            for comb in combs:
                temp_feature.append(corr_feature(df_EMG_filt[comb[0]].values,df_EMG_filt[comb[1]].values))
            temp_feature = pd.DataFrame(temp_feature)
            # THE EMG FEATURES NEED TO BE 
            df_features = df_features.append(temp_feature,ignore_index=True)
            # 120 additional features = 248 total

        # EXTRACT IMU DATA
        df_IMU = pd.read_csv(path+IMU_files[ix],header=0,index_col=0)
        # # save as csv files
        # df_EMG_movavg.to_csv(save_path+'movavg_files/' + file[:-4] + "_movavg.csv")
        for ix,time_feature in enumerate(time_features_IMU):
            temp_feature = pd.DataFrame(list(df_IMU.apply(lambda s: time_feature(s.values,148.148),axis=0).values))
            df_features = df_features.append(temp_feature,ignore_index=True)
            # + 72 features = 320 total

        # save expert data
        df_features.to_csv(save_path+'expert_files/' + file[:-4] + "_expert.csv",index=False,header=False)