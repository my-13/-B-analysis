from globals import *

def test_train_split(participantID,experiment_types,etype,n_templates=3):
    # random.seed(N_seed)
    files_path = temp_path +participantID+"\\"
    save_path = models_path + participantID + "\\"

    # make new directory
    new_foldername = etype
    if not os.path.isdir(save_path+new_foldername):
        os.mkdir(save_path+new_foldername)

    experiment_types['motionlabel'] = experiment_types['motion'].apply(lambda motion: label_motions(motion,motions)).astype(np.float32)
    experiment_types['pname_EMG'] = files_path+"movavg_files/" # full path name for EMG
    experiment_types['fname_EMG'] = experiment_types['fname_EMG'].apply(lambda fname: convert_fname(fname,'movavg'))
    experiment_types['pname_IMU'] = files_path+"IMU_extract/" # full path name for EMG
    experiment_types['fname_IMU'] = experiment_types['fname_IMU'].apply(lambda fname: convert_fname(fname,'IMU_extract'))
    
    if etype!='rotate':
        # define test-train datasets
        types = experiment_types["motion"].unique()
        experiment_types['template'] = [False for row in range(len(experiment_types))]
        for ty in types:
            experiment_types.loc[experiment_types[experiment_types['motion'] == ty].sample(n_templates).index,'template'] = True
        training = experiment_types.loc[experiment_types['template']==True]
        training = training.drop(labels='template',axis=1)
        training.to_csv(save_path+new_foldername+"/"+participantID+"_"+etype+"_data_location_training_"+str(n_templates)+".csv",index=False,header=True)

        testing = experiment_types.loc[experiment_types['template']==False]
        testing = testing.drop(labels='template',axis=1)
        testing["to_test"] = False
        testing.loc[testing.sample(1).index,"to_test"] = True
        testing = testing.loc[testing['to_test']==True] # take this out if needed 
        testing.to_csv(save_path+new_foldername+"/"+participantID+"_"+etype+"_data_location_testing_"+str(n_templates)+".csv",index=False,header=True)
    else: # get values for analogous
        experiment_types["to_test"] = False
        types = experiment_types["motion"].unique()
        for ty in types:
            experiment_types.loc[experiment_types[experiment_types['motion'] == ty].sample(1).index,'to_test'] = True
        experiment_types = experiment_types.loc[experiment_types['to_test']==True] # take this out if needed
        experiment_types.to_csv(save_path+new_foldername+"/"+participantID+"_"+etype+"_data_location_analogous_"+str(n_templates)+".csv",index=False,header=True)
        
        # for vanillaNN
        for ty in types:
            df_ex = experiment_types.loc[(experiment_types["motion"]==ty) & (experiment_types["to_test"]==True)]
            df_ex.to_csv(save_path+new_foldername+"/"+participantID+"_"+etype+"_data_location_analogous_"+ty+"_"+str(n_templates)+".csv",index=False,header=True)

def get_standardized_files(pID,n_templates,isTrain):
    # save_path = temp_path + pID + "//movavg_files//" 
    files_path = temp_path +pID+"\\"
    experiment_types = pd.read_csv(temp_path[:-24] +'fnames/'+pID+'_fnames.csv',header=0,index_col=0)
    experiment_types = experiment_types.loc[(experiment_types['etype']=='standardized') & (experiment_types['motion'].isin(standard_motions))]
    experiment_types['motionlabel'] = experiment_types['motion'].apply(lambda motion: label_motions(motion,standard_motions)).astype(np.float32)
    types = experiment_types["motion"].unique()
    if isTrain:
        experiment_types['template'] = [False for row in range(len(experiment_types))]
        for ty in types:
            experiment_types.loc[experiment_types[experiment_types['motion'] == ty].sample(n_templates).index,'template'] = True
        experiment_types = experiment_types.loc[experiment_types['template']==True]
        # experiment_types = experiment_types.drop(labels='template',axis=1)
    else:
        experiment_types["to_test"] = False
        experiment_types.loc[experiment_types.sample(1).index,"to_test"] = True
        experiment_types = experiment_types.loc[experiment_types['to_test']==True]  
    experiment_types['pname_EMG'] = files_path+"movavg_files/" # full path name for EMG
    experiment_types['vanilla_path'] = files_path+"movavg_files/" # full path name for EMG
    experiment_types['pID'] = pID
    experiment_types['fname_EMG'] = experiment_types['fname_EMG'].apply(lambda fname: convert_fname(fname,"movavg"))
    
    return experiment_types

def convert_fname(fname,folder):
    return fname[:-4]+'_'+folder+'.csv'

def label_motions(motion,labels):
    if motion not in labels:
        if motion=='shrink':
            return int(8)
        elif motion=='enlarge':
            return int(9)
        elif motion=='normal' or motion=='frequency' or motion == 'range-of-motion':
            return int(2)
        else:
            print(motion, ' could not be labeled')
    else:
        return labels.index(motion)

# DOLLAR ALGORITHMS
def dollar(templates,test_data,spath_EMG):
    # define template data
    templates_for_dollar = []
    for t in range(len(templates)):
        fname_EMG = templates.fname_EMG.iloc[t][:-11]+'_dollar.csv'
        if spath_EMG==None:
            templates_for_dollar.append(([templates.motion.iloc[t],
                                        [pd.read_csv(templates.pname_EMG.iloc[t][:-13]+"dollar_files/"+fname_EMG,header=None,index_col=False).values.T.tolist()]]))
        else:
            templates_for_dollar.append(([templates.motion.iloc[t],
                                        [pd.read_csv(spath_EMG+fname_EMG,header=None,index_col=False).values.T.tolist()]]))

    templates_for_dollar = np.asarray(templates_for_dollar,dtype=object)
    mdoll = Mdollar(templates_for_dollar)
    del templates_for_dollar
    gc.collect()

    # get test data classification
    test_data['classified_motion'] = ['' for e in range(len(test_data))]

    for t in range(len(test_data)):
        fname_EMG = test_data.fname_EMG.iloc[t][:-11]+'_dollar.csv'
        if spath_EMG==None:
            test_data.loc[t,'classified_motion'] = mdoll.get_gesture(
                                        [pd.read_csv(test_data.pname_EMG.iloc[t][:-13]+"dollar_files/"+fname_EMG,header=None,index_col=False).values.T.tolist()])
        else:
            test_data.loc[t,'classified_motion'] = mdoll.get_gesture(
                                        [pd.read_csv(spath_EMG+fname_EMG,header=None,index_col=False).values.T.tolist()])

    # return whether accurate (1) or not accurate (0)
    test_accuracy = accuracy_score(test_data['motion'],test_data['classified_motion'],normalize=False)
    return mdoll,test_accuracy

def dollar_normal(pID,etype,n_templates=3):
    if not os.path.isdir(models_path + pID+"/"+etype):
        os.mkdir(models_path+pID+"/"+etype)
    classifier_save = models_path+pID+"/"+etype+'/'+pID +"_" +  etype + "_"
    templates = pd.read_csv(classifier_save+"data_location_training_"+str(n_templates)+".csv",
                                    header=0)
    # define test data
    test_data = pd.read_csv(classifier_save+"data_location_testing_"+str(n_templates)+".csv",
                                    header=0)
    spath_EMG = temp_path + pID +"/dollar_files/"
    doll,test_accuracy = dollar(templates,test_data,spath_EMG)
    # print("dollar ",pID,"train-test accuracy: ", train_accuracy,test_accuracy)
    # test_data.to_csv(classifier_save + 'dollar_test_classifications.csv')
    # templates.to_csv(classifier_save + 'dollar_train_classifications.csv')

    # save dollar model
    with open(classifier_save+"dollar_model_"+str(n_templates)+".pickle","wb") as handle:
        pickle.dump(doll,handle,protocol=pickle.HIGHEST_PROTOCOL)
    del doll
    gc.collect()

    return test_accuracy

def dollar_rotate(pID,n_templates,etype='rotate'):
    classifier_save = models_path+pID+"/"+etype+'/'+pID + "_" + etype + "_"
    if not os.path.isdir(models_path+pID+"/"+etype):
        os.mkdir(models_path+pID+"/"+etype)
    with open(models_path+pID+"/experimenter-only/"+pID+"_experimenter-only_dollar_model_"+str(n_templates)+".pickle","rb") as handle:
        mdoll = pickle.load(handle)
    test_data = pd.read_csv(classifier_save+"data_location_analogous_"+str(n_templates)+".csv",
                                    header=0)
    
   # get test data classification
    test_data['classified_motion'] = ['' for e in range(len(test_data))]
    spath_EMG = temp_path + pID +"/dollar_files"+"/"
    for t in range(len(test_data)):
        # only test 1 file in test
        # if test_data.to_test.iloc[t]: 
        # spath_EMG = test_data.pname_EMG.iloc[t]
        # spath_IMU = test_data.pname_IMU.iloc[t]
        fname_EMG = test_data.fname_EMG.iloc[t][:-11]+'_dollar.csv'
        # fname_IMU = test_data.fname_IMU.iloc[t]

        # test_data.loc[t,'classified_motion'] = mdoll.get_gesture(
        #                               [pd.read_csv(spath_EMG+fname_EMG,header=0,index_col=0).values.T.tolist(),
        #                               pd.read_csv(spath_IMU+fname_IMU,header=0,index_col=0).values.T.tolist()])
        test_data.loc[t,'classified_motion'] = mdoll.get_gesture(
                                      [pd.read_csv(spath_EMG+fname_EMG,header=None,index_col=False).values.T.tolist()])

    normal_accuracy = obtain_values(test_data,'normal')
    fast_accuracy = obtain_values(test_data,'frequency')
    large_accuracy = obtain_values(test_data,'range-of-motion')
    del mdoll
    gc.collect()
    return normal_accuracy,fast_accuracy,large_accuracy

def obtain_values(test_data,type,dtype="str"):
    classifications = test_data.loc[test_data['motion']==type]['classified_motion']
    # if len(classifications) != 10:
    #     print('classifications for ',type,' not accurate')
    if dtype=="str":
        real_label = np.squeeze(np.matlib.repmat(['rotate'],len(classifications),1))
        return accuracy_score([real_label],classifications.values,normalize=False)
    elif dtype=="int":
        real_label = [np.squeeze(np.matlib.repmat([2],len(classifications),1))]
        return accuracy_score(real_label,classifications.astype(int).values,normalize=False)

# VANILLA NN
class BiosignalDataset(Dataset):
    """Biosignal dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # if testing only pull out the file to test
        temp = pd.read_csv(csv_file)
        # if csv_file[-11:-4] == "testing":
        #     temp = temp.loc[temp["to_test"]==True]
        # else:
        #     pass;
        self.biosignals = temp
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.biosignals)

    def __getitem__(self, idx):

        # if self.root_dir==None:
        #     try:
        #         biosignal_path = os.path.join(self.biosignals.iloc[idx,6][:-13],"vanillaNN_files//",self.biosignals.iloc[idx,0][:-4]+'_vanillaNN.csv')
        #         label = self.biosignals.iloc[idx,5]
        #     except:
        #         print(self.biosignals.iloc[idx])
        # else:
        if self.root_dir==None:
            biosignal_path = os.path.join(self.biosignals.pname_EMG.iloc[idx][:-13]+"dollar_files/",
                                          self.biosignals.fname_EMG.iloc[idx][:-11]+'_dollar.csv')
        else:
            biosignal_path = os.path.join(self.root_dir,self.biosignals.fname_EMG.iloc[idx][:-11]+'_dollar.csv')
        label = self.biosignals.motionlabel.iloc[idx]
        data = pd.read_csv(biosignal_path,header=None,index_col=None).values.T.tolist()
        data = torch.tensor(data)
        return data, label
    
# a Vanilla NN with one 64 node hidden layer 
class NeuralNetwork(nn.Module):
    # FFNN CODE FOR 2024 UIST SUBMISSION
    # def __init__(self):
    #     super().__init__()
    #     self.flatten = nn.Flatten()
    #     self.linear_relu_stack = nn.Sequential(
    #         nn.Linear(5632, 64),
    #         nn.ReLU(),
    #         # nn.Dropout(0.8),
    #         nn.Linear(64, 10),
    #     )

    # def forward(self, x):
    #     x = self.flatten(x)
    #     logits = self.linear_relu_stack(x)
    #     return logits
    
    # CNN + LSTM CODE FOR CHI SUBMISSION

    def __init__(self):
        hidden_size = 64
        tlen = 62
        super().__init__()

        # THIS IS CNN LAYERS TO EXTRACT FEATURES
        self.conv1 = nn.Conv1d(in_channels=88,out_channels=88,kernel_size=2,stride=1)
        self.pool = nn.MaxPool1d(kernel_size=2,stride=1)

        # THIS IS THE DENSE LAYER BETWEEN CNN AND LSTM
        self.dense = nn.Linear(88,88)

        self.relu = nn.ReLU()

        # THIS IS LSTM LAYER
        self.lstm = nn.LSTM(input_size=88, hidden_size=hidden_size, num_layers=2, batch_first=True,dropout=0.8)
        self.flatten = nn.Flatten()

        # this is FFNN for the classification
        self.linear_relu_stack = nn.Sequential( 
            nn.Linear(int(hidden_size*tlen), 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        x = self.pool(self.conv1(x)) # CNN layer + pooling
        x = torch.swapaxes(x,1,2)
        x = self.relu(self.dense(x)) # fully-connected? dense layer
        x, _ = self.lstm(x) # 2 LSTM layers
        x = self.flatten(x) 
        logits = self.linear_relu_stack(x) # FFNN/dense layer for classification
        return logits
    
def get_device():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    return device

def vanillaNN_normal(pID,etype,n_templates=3):
    if not os.path.isdir(models_path+pID+"/"+etype):
        os.mkdir(models_path+pID+"/"+etype)

    num_epochs = 20
    batch_size = 10
    save_path = temp_path + pID +"/dollar_files"+"/"
    classifier_save = models_path+pID+"/"+etype+'/'+pID + "_" + etype + "_"
    training_data = BiosignalDataset(
        csv_file = classifier_save+"data_location_training_"+str(n_templates)+".csv",
        root_dir=save_path
    )
    test_data = BiosignalDataset(
        csv_file = classifier_save +"data_location_testing_"+str(n_templates)+".csv",
        root_dir=save_path
    )    
    # model, test_data_classifications,train_data_classifications,train, test = vanillaNN(training_data,test_data,batch_size,num_epochs)
    model, test_acc = vanillaNN(training_data,test_data,batch_size,num_epochs)

    # train_loss,train_acc = train
    # val_loss,val_acc = test
    # save data
    # torch.save(model.state_dict(),classifier_save+'vanillaNN_model_states'+'.pth')
    torch.save(model, classifier_save+'vanillaNN_model_'+str(n_templates)+'.pt')
    del model
    gc.collect()
    # test_data_classifications.to_csv(classifier_save + 'vanillaNN_test_classifications'+'.csv')
    # train_data_classifications.to_csv(classifier_save+'vanillaNN_train_classifications'+'.csv')
    return test_acc#np.asarray([train_loss,train_acc]), np.asarray([val_loss,val_acc])

def vanillaNN(training_data,test_data,batch_size,num_epochs,model=None):
    if model==None:
        model = NeuralNetwork().to(get_device())
        model.zero_grad()

    # define loss and optimizer
    criterion = nn.CrossEntropyLoss()  # categorical cross entropy loss_fn
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    # train_loss_history = []
    # train_acc_history = []
    # val_loss_history = []
    # val_acc_history = []
    
    # Loop through the number of epochs
    for epoch in range(num_epochs):
        # train_loss = 0.0
        # train_acc = 0.0
        # val_loss = 0.0
        # val_acc = 0.0
    
        # set model to train mode
        model.train()
        # iterate over the training data
        for inputs, labels in train_loader:
            inputs = inputs.to(torch.float).cuda()
            labels = labels.to(torch.int64).cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            #compute the loss
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # # increment the running loss and accuracy
            # train_loss += loss.item()
            # train_acc += (outputs.argmax(1) == labels).sum().item()
    
        # # calculate the average training loss and accuracy
        # train_loss /= len(train_loader)
        # train_loss_history.append(train_loss)
        # train_acc /= len(train_loader.dataset)
        # train_acc_history.append(train_acc)
    
        # # set the model to evaluation mode
        # model.eval()
        # with torch.no_grad():
        #     for inputs, labels in test_loader:
        #         inputs = inputs.to(torch.float).cuda()
        #         labels = labels.to(torch.int64).cuda()
        #         outputs = model(inputs)
        #         loss = criterion(outputs, labels)
        #         val_loss += loss.item()
        #         val_acc += (outputs.argmax(1) == labels).sum().item()
    
        # # calculate the average validation loss and accuracy
        # val_loss /= len(test_loader)
        # val_loss_history.append(val_loss)
        # val_acc /= len(test_loader.dataset)
        # val_acc_history.append(val_acc)

    # final evaluation of test and train sets
    # train_loader = DataLoader(training_data, batch_size=len(training_data), shuffle=False)
    test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)
    model.eval()
    with torch.no_grad():
        test_val_acc = 0.0
        for inputs, labels in test_loader:
            inputs = inputs.to(torch.float).cuda()
            labels = labels.to(torch.int64).cuda()
            outputs = model(inputs)
            test_val_acc += (outputs.argmax(1) == labels).sum().item()
    # test_val_acc /= len(test_loader.dataset)
    # test_data_classifications = pd.DataFrame(np.array([labels.cpu().numpy(),outputs.argmax(1).cpu().numpy()]).T,columns=['motion','classified motion'])
    
    # model.eval()
    # with torch.no_grad():
    #     train_val_acc = 0.0
    #     for inputs, labels in train_loader:
    #         inputs = inputs.to(torch.float).cuda()
    #         labels = labels.to(torch.int64).cuda()
    #         outputs = model(inputs)
    #         train_val_acc += (outputs.argmax(1) == labels).sum().item()
    # train_val_acc /= len(train_loader.dataset)
    # train_data_classifications = pd.DataFrame(np.array([labels.cpu().numpy(),outputs.argmax(1).cpu().numpy()]).T,columns=['motion','classified motion'])
    # print(f'Vanilla NN final train-test accuracy, {train_val_acc:.4f},{test_val_acc:.4f}')
    
    return model, test_val_acc#np.asarray([train_loss,train_acc]), np.asarray([val_loss,val_acc])

def vanillaNN_rotate(pID,n_templates):
    etype="rotate"
    if not os.path.isdir(models_path+pID+"/"+etype):
        os.mkdir(models_path+pID+"/"+etype)
    classifier_save = models_path+pID+"/"+etype+'/'+pID + "_" + etype + "_"
    save_path = temp_path + pID +"/dollar_files"+"/"
    # load model
    model = torch.load(models_path+pID+"/experimenter-only/"+pID+"_experimenter-only_vanillaNN_model_"+str(n_templates)+".pt")
    types = ["normal","frequency","range-of-motion"]
    acc = np.zeros((3,))
    for ix,ty in enumerate(types):
        # load test data
        test_data = BiosignalDataset(
            csv_file = classifier_save +"data_location_analogous_"+ty+"_"+str(n_templates)+".csv",
            root_dir=save_path
        )
        model.eval()

        test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)
        with torch.no_grad():
            test_val_acc = 0.0
            for inputs, labels in test_loader:
                inputs = inputs.to(torch.float).cuda()
                labels = labels.to(torch.int64).cuda()
                outputs = model(inputs)
                test_val_acc += (outputs.argmax(1) == labels).sum().item()
        # test_val_acc /= len(test_loader.dataset)
        # test_data_classifications = pd.DataFrame(np.array([labels.cpu().numpy(),outputs.argmax(1).cpu().numpy()]).T,columns=['motion','classified motion'])
        # test_data_classifications.to_csv(classifier_save + 'vanillaNN_test_classifications'+'.csv')
        # print("vanillaNN  ",pID," accuracy: ",test_val_acc)
        acc[ix] = test_val_acc
    del model
    gc.collect()
    return acc

#  EXPERT CODE
def expert_normal(pID,etype,n_templates=3):
    if not os.path.isdir(models_path+pID+"/"+etype):
        os.mkdir(models_path+pID+"/"+etype)
    read_path = temp_path+pID+"/"+"expert_files/"
    classifier_save = models_path+pID+"/"+etype+'/'+pID + "_" + etype + "_"
    

    # define test and train data
    train_data = pd.read_csv(classifier_save+"data_location_training_"+str(n_templates)+".csv",
                                    header=0)
    test_data = pd.read_csv(classifier_save+"data_location_testing_"+str(n_templates)+".csv",
                                    header=0)
    # test_data = test_data.reset_index()

    # train_data,test_data,clf,train_accuracy,test_accuracy = expert(train_data,test_data,read_path)
    clf,test_accuracy = expert(train_data,test_data,read_path)

    # test_data.to_csv(classifier_save+ 'expert_test_classifications.csv')
    # train_data.to_csv(classifier_save+ 'expert_train_classifications.csv')

    # save expert features model
    with open(classifier_save+"expert_model_"+str(n_templates)+".pickle","wb") as handle:
        pickle.dump(clf,handle,protocol=pickle.HIGHEST_PROTOCOL)
    # print("expert ",pID,"train-test accuracy: ",train_accuracy,test_accuracy)
    return test_accuracy# train_accuracy,test_accuracy

def expert(train_data,test_data,read_path):
    # train model
    X = []
    Y = []

    for t in range(len(train_data)):
        fname = train_data.fname_EMG.iloc[t]
        if read_path == None:
            X.append(np.asarray(np.squeeze(np.asarray(pd.read_csv(train_data.pname_EMG.iloc[t][:-13]+"expert_files/" + fname[:-10]+"expert.csv",index_col=None,header=None))).tolist()))
        else:
            # temp_file = np.squeeze(np.asarray(pd.read_csv(read_path + fname[:-10]+"expert.csv",index_col=None,header=None)))
            X.append(np.asarray(np.squeeze(np.asarray(pd.read_csv(read_path + fname[:-10]+"expert.csv",index_col=None,header=None))).tolist()))
        Y.append(train_data.motionlabel.iloc[t])
    X = np.asarray(X)
    Y = np.asarray(Y)
    clf = svm.SVC(kernel='linear') # Linear Kernel
    clf.fit(X,Y)

    # obtain train accuracy
    train_data['classified_motion'] = ['' for e in range(len(train_data))]
    
    # for t in range(len(train_data)):
    #     fname = train_data.fname_EMG.iloc[t]
    #     temp_file = np.squeeze(np.asarray(pd.read_csv(read_path + fname[:-10]+"expert.csv",index_col=None,header=None)))
    #     train_data.loc[t,'classified_motion'] = clf.predict(np.asarray(temp_file.tolist()).reshape(1, -1))

    # obtain test accuracy
    test_data['classified_motion'] = ['' for e in range(len(test_data))]
    for t in range(len(test_data)):
        fname = test_data.fname_EMG.iloc[t]
        if read_path == None:
           temp_file = np.squeeze(np.asarray(pd.read_csv(test_data.pname_EMG.iloc[t][:-13]+"expert_files/" + fname[:-10]+"expert.csv",index_col=None,header=None))) 
        else:
            temp_file = np.squeeze(np.asarray(pd.read_csv(read_path + fname[:-10]+"expert.csv",index_col=None,header=None)))
        test_data.loc[t,'classified_motion'] = clf.predict(np.asarray(temp_file.tolist()).reshape(1, -1))

    # return whether accurate (1) or not accurate (0)
    test_accuracy = accuracy_score(test_data['motionlabel'].astype(int).values,
                                    test_data['classified_motion'].astype(int).values,
                                    normalize=False)
    return clf,test_accuracy#train_data,test_data,clf,train_accuracy,test_accuracy

def expert_rotate(pID,n_templates):
    read_path = temp_path+pID+"/"+"expert_files/"
    etype = "rotate"
    if not os.path.isdir(models_path+pID+"/"+etype):
        os.mkdir(models_path+pID+"/"+etype)
    classifier_save = models_path+pID+"/"+etype+'/'+pID + "_"+ etype + "_"
    with open(models_path+pID+"/experimenter-only/"+pID+"_experimenter-only_expert_model_"+str(n_templates)+".pickle","rb") as handle:
        clf = pickle.load(handle)
    save_path = models_path + pID +"/rotate/"
    test_data = pd.read_csv(save_path+pID+"_rotate_data_location_analogous_"+str(n_templates)+".csv",
                                header=0)
    # test_data = test_data.reset_index()
    # get test data classification
    # test_data['classified_motion'] = [1000 for e in range(len(test_data))]

    for t in range(len(test_data)):
        # if test_data.to_test.iloc[t]:
        fname = test_data.fname_EMG.iloc[t]
        temp_file = np.squeeze(np.asarray(pd.read_csv(read_path + fname[:-10]+"expert.csv",index_col=None,header=None)))
        test_data.loc[t,'classified_motion'] = clf.predict(np.asarray(temp_file.tolist()).reshape(1, -1))
        # else:
        #     pass;
    # compute accuracies, should all be rotate
    normal_accuracy = obtain_values(test_data,'normal',"int")
    fast_accuracy = obtain_values(test_data,'frequency',"int")
    large_accuracy = obtain_values(test_data,'range-of-motion',"int")
    # print("expert  ",pID," overall,normal,fast,large accuracy: ",overall_accuracy,normal_accuracy,fast_accuracy,large_accuracy)
        
    # test_data.to_csv(classifier_save + 'expert_test_classifications.csv')
    return normal_accuracy,fast_accuracy,large_accuracy

def train_standardized(n_templates,algoType,pIDs_test):
    etype = 'standardized'
    classifier_save = models_path+pIDs_test[0] + "/" + etype + "/" + pIDs_test[0] + "_" + etype + "_"
    if not os.path.isdir(temp_path+etype):
        os.mkdir(temp_path+etype)  
    train_data = pd.DataFrame()
    if algoType == "dollar":
        fpath = "dollar_files"
    elif algoType == "vanillaNN":
        fpath = 'vanillaNN_files'
    elif algoType == "expert":
        fpath = "expert_files"


    train_data = pd.read_csv(classifier_save+"data_location_training_"+str(n_templates)+".csv",
                                    header=0)
    test_data = pd.read_csv(classifier_save+"data_location_testing_"+str(n_templates)+".csv",
                                    header=0)

    if algoType=='dollar':
        _,test_accuracy = dollar(train_data,test_data,spath_EMG=None)
        # print("dollar ",pID,"train-test accuracy: ", train_accuracy,test_accuracy)

    if algoType=='vanillaNN':
        training_data = BiosignalDataset(
            csv_file = classifier_save+"data_location_training_"+str(n_templates)+".csv",
            root_dir=None
        )
        testing_data = BiosignalDataset(
            csv_file = classifier_save+"data_location_testing_"+str(n_templates)+".csv",
            root_dir=None
        )

        num_epochs = 20
        batch_size = 10
        _,  test_accuracy = vanillaNN(training_data,testing_data,batch_size,num_epochs)

        torch.cuda.empty_cache() # clear gpu

    elif algoType=='expert':
        _, test_accuracy = expert(train_data,test_data,read_path=None)

    return test_accuracy