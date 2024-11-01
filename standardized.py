from globals import *
from algorithms1 import get_standardized_files, train_standardized
algoTypes = ["dollar","vanillaNN","expert"]

def run_standardized(n_templates):

    print("running ",n_templates, " templates")
    df_accuracy = pd.DataFrame()
    for ix in range(100):
        eval_standardized(n_templates)
        df_accuracy = df_accuracy.append(eval_standardized(n_templates),ignore_index=True)
        if ix % 10 == 0:
            print("on ", ix)
        gc.collect()

    df_standardized = pd.DataFrame()
    # # FOR PERSONALIZED 
    for pID in pIDs_standardized:
        for algoType in algoTypes:
            temp = df_accuracy.loc[(df_accuracy["pID"]==pID) & (df_accuracy["algoType"]==algoType)].reset_index()
            accuracy = np.asarray([[pID,algoType,n_templates,temp.test.mean(),temp["ptype"][0]]])
            df_temp = pd.DataFrame(accuracy,columns=['pID','algoType','nTemplates','test_mean',"ptype"])
            df_standardized = df_standardized.append(df_temp,ignore_index=True)

    df_standardized['test_mean'] = df_standardized.test_mean.astype(float)

    # SAVE THE CVS
    df_standardized.to_csv(models_path + 'standardized_results_'+str(n_templates)+'.csv',index=False)

def eval_standardized(n_templates):
    df_accuracy = pd.DataFrame()
    # data = []
    # standardized
    for ix,pID in enumerate(pIDs_standardized):
        # get the files
        pIDs_train = np.delete(np.asarray(pIDs_standardized),ix)
        get_test_train_split(n_templates,pIDs_train,[pID])

        # train and test the algorithms
        test_dollar = train_standardized(n_templates,"dollar",[pID])
        test_vanilla = train_standardized(n_templates,"vanillaNN",[pID])
        test_expert = train_standardized(n_templates,"expert",[pID])

        accuracy = np.asarray([
                            [test_vanilla,'vanillaNN'],
                            [test_dollar,'dollar'],
                            [test_expert,'expert']
                            ])
        df_temp = pd.DataFrame(accuracy,columns=['test','algoType'])
        
        if pID[1]=='1':
            df_temp['ptype'] = 'disabled'
        else:
            df_temp['ptype'] = 'non-disabled'
        df_temp['pID'] = pID

        df_accuracy = df_accuracy.append(df_temp,ignore_index=True)
    df_accuracy['test'] = df_accuracy.test.astype(float)
    return df_accuracy

def get_test_train_split(n_templates,pIDs_train,pIDs_test):
    # obtain train features
    etype = "standardized"
    if not os.path.isdir(models_path + pIDs_test[0] + "//standardized//"):
        os.mkdir(models_path + pIDs_test[0] + "//standardized")
    save_path = models_path + pIDs_test[0] + "//standardized//" + pIDs_test[0] + "_"
    train_data = pd.DataFrame()
    for pID in pIDs_train:
        train_data = train_data.append(get_standardized_files(pID,n_templates,isTrain=True),ignore_index=True)
    
    test_data = pd.DataFrame()
    # obtain test feature
    for pID in pIDs_test:
        test_data = test_data.append(get_standardized_files(pID,n_templates,isTrain=False),ignore_index=True)
    test_data.to_csv(save_path +etype+"_data_location_testing_"+str(n_templates)+".csv",index=False,header=True)
    train_data.to_csv(save_path +etype+"_data_location_training_"+str(n_templates)+".csv",index=False,header=True)

