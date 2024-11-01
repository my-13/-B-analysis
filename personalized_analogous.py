from globals import *
from algorithms1 import test_train_split,dollar_normal, dollar_rotate, vanillaNN_normal,vanillaNN_rotate,expert_normal,expert_rotate


def run_personalized_analogous(n_templates):
    algoTypes = ["dollar","vanillaNN","expert"]
    df_personalized = pd.DataFrame()
    df_analogous = pd.DataFrame()
    # template = 1 through 9

    # ONLY DO IF TEMPLATE 1 IS ALREADY RUN
    # df_personalized = pd.read_csv(models_path + 'personalized_results.csv',header=0)
    # df_analogous = pd.read_csv(models_path + 'analogous_results.csv',header=0)
    print("running ",n_templates, " templates")
    df_accuracy = pd.DataFrame()
    df_rotate = pd.DataFrame()
    for ix in range(100):
        # PERSONALIZED
        get_test_train_split(n_templates) 
        temp = eval_personalized(n_templates)
        df_accuracy = df_accuracy.append(temp,ignore_index=True)

        # ALSO RUN ANALOGOUS HERE USING THE EXISTING MODELS
        temp = eval_analogous(n_templates)
        df_rotate = df_rotate.append(temp,ignore_index=True)
        if ix % 10 == 0:
            print("on ", ix)
        gc.collect()

    # # FOR PERSONALIZED 
    for pID in pIDs_personalized:
        for algoType in algoTypes:
            temp = df_accuracy.loc[(df_accuracy["pID"]==pID) & (df_accuracy["algoType"]==algoType)].reset_index()
            accuracy = np.asarray([[pID,algoType,n_templates,temp.test.mean(),temp["ptype"][0]]])
            df_temp = pd.DataFrame(accuracy,columns=['pID','algoType','nTemplates','test_mean',"ptype"])
            df_personalized = df_personalized.append(df_temp,ignore_index=True)

    # FOR ANALOGOUS 
    for pID in pIDs_analogous:
        for algoType in algoTypes:
            types = df_rotate["type"].unique()
            for ty in types:
                temp = df_rotate.loc[(df_rotate["pID"]==pID) & (df_rotate["algoType"]==algoType) & 
                                    (df_rotate["type"]==ty)].reset_index()
                accuracy = np.asarray([[pID,algoType,ty,n_templates,temp.test.mean(),temp["ptype"][0]]])
                df_temp = pd.DataFrame(accuracy,columns=['pID','algoType','gestureType','nTemplates','test_mean',"ptype"])
                df_analogous = df_analogous.append(df_temp,ignore_index=True)
    del df_rotate,df_accuracy
    gc.collect()
    # SAVE THE CVS TEMPORARILY
    # df_personalized.to_csv(models_path + 'personalized_results.csv',index=False)
    # df_analogous.to_csv(models_path + 'analogous_results.csv',index=False)
    # break;

    # convert str to float
    df_analogous['test_mean'] = df_analogous.test_mean.astype(float)
    df_personalized['test_mean'] = df_personalized.test_mean.astype(float)

    # SAVE THE CVS
    df_personalized.to_csv(models_path + 'personalized_results_'+str(n_templates)+'.csv',index=False)
    df_analogous.to_csv(models_path + 'analogous_results_'+str(n_templates)+'.csv',index=False)

def eval_personalized(n_templates):
    df_accuracy = pd.DataFrame()
    # data = []
    etype = 'experimenter-only'
    for pID in pIDs_personalized:
        test_dollar = dollar_normal(pID,etype,n_templates=n_templates)
        test_vanilla = vanillaNN_normal(pID,etype,n_templates=n_templates)
        test_expert = expert_normal(pID,etype,n_templates=n_templates)

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

def eval_analogous(n_templates):
    df_rotate_accuracy = pd.DataFrame()
    for pID in pIDs_analogous:
        accuracies_dollar = dollar_rotate(pID,n_templates=n_templates)
        accuracies_expert = expert_rotate(pID,n_templates=n_templates)
        accuracies_vanillaNN = vanillaNN_rotate(pID,n_templates=n_templates)

        accuracy = np.asarray([
                                [accuracies_vanillaNN[0],accuracies_vanillaNN[1],accuracies_vanillaNN[2],'vanillaNN'],
                                [accuracies_dollar[0],accuracies_dollar[1],accuracies_dollar[2],'dollar'],
                                [accuracies_expert[0],accuracies_expert[1],accuracies_expert[2],'expert']
                                ])
        df_temp = pd.DataFrame(accuracy,columns=['normal','fast','large','algoType'])
        if pID[1]=='1':
            df_temp['ptype'] = 'disabled'
        else:
            df_temp['ptype'] = 'non-disabled'
        df_temp['pID'] = pID
        df_rotate_accuracy = df_rotate_accuracy.append(df_temp)
    df_rotate_accuracy['normal'] = df_rotate_accuracy.normal.astype(float)
    df_rotate_accuracy['fast'] = df_rotate_accuracy.fast.astype(float)
    df_rotate_accuracy['large'] = df_rotate_accuracy.large.astype(float)
    df_rotate_accuracy = pd.melt(df_rotate_accuracy,id_vars=["pID","algoType","ptype"],var_name="type",
                                 value_name="test",ignore_index=True)
    return df_rotate_accuracy

def get_test_train_split(n_templates):
    etype = "experimenter-only"
    for pID in pIDs_personalized:
        experiment_types = pd.read_csv(temp_path[:-24]+'fnames/'+pID+'_fnames.csv',header=0,index_col=0)
        experiment_types = experiment_types.loc[experiment_types['etype']=='experimenter-defined']
        test_train_split(pID,experiment_types,etype=etype,n_templates=n_templates)

    # rotate
    etype = "rotate"
    for pID in pIDs_analogous:
        experiment_types = pd.read_csv(temp_path[:-24]+'fnames/'+pID+'_fnames.csv',header=0,index_col=0)
        experiment_types = experiment_types.loc[(experiment_types['etype']=='rehab') & (experiment_types['motion']!="endurance")]
        test_train_split(pID,experiment_types,etype=etype,n_templates=n_templates)

# if __name__ == "__main__":
#     main()
        
def run_analogous(n_templates):
    times = []
    for ix in range(100):
        t = time.time()
        for pID in [pIDs_analogous[18]]:
            dollar_rotate(pID,n_templates=n_templates)
        # if ix % 10 == 0:
        #     print("on ", ix)
        times.append(time.time()-t)
    return np.asarray(times)

