import numpy as np
import pandas as pd


# Define the function to read the data 
def read_data(data_path):
    pd0=pd.read_csv(data_path+"/MLData2022.csv", sep=',')
    pd1=pd.read_csv(data_path+"/evalRuns.csv", sep=',')
    pd2=pd.read_csv(data_path+"/testRuns.csv", sep=',')

    pd1['classify_ae']=np.ones(pd1.shape[0])
    pd1['anomaly_new']=np.ones(pd1.shape[0])
    pd1['anomaly_new_adapt']=np.ones(pd1.shape[0])

    pd2['classify_ae']=np.zeros(pd2.shape[0])
    pd2['anomaly_new']=np.zeros(pd2.shape[0])
    pd2['anomaly_new_adapt']=np.zeros(pd2.shape[0])
    

    #getting the distances
    dist_q=np.array(pd2[['D_euc_vec_1', 'D_euc_vec_2', 'D_dtw_vec_1', 'D_dtw_vec_2']])
    dist_nq=np.array(pd1[['D_euc_vec_1', 'D_euc_vec_2', 'D_dtw_vec_1', 'D_dtw_vec_2']])
    dist_2022=np.array(pd0[['Var6_1', 'Var6_2', 'Var7_1', 'Var7_2']])
    print(dist_nq.shape)
    
    id=dist_nq[:, 0]!=0
    dist_nq=dist_nq[id, :]
    print(dist_nq.shape)
    

    # getting the ground truth
    gt_q=np.array(pd2['classify_ae'])
    gt_nq=np.array(pd1['classify_ae'])[id]
    print(gt_nq.shape)
    gt_2022=np.array(pd0[['Var8']])


    #split the data
    train =np.concatenate([dist_q, dist_nq, dist_2022[0:100000, :]], axis=0)
    test=dist_2022[100000:, :]

    gt_2022=gt_2022.reshape(gt_2022.shape[0],)
    print(f"{gt_q.shape}, {gt_nq.shape}, {gt_2022.shape}")
    gt_2022[gt_2022!=0]=1

    gt_train=np.concatenate([gt_q, gt_nq, gt_2022[0:100000]], axis=0)
    gt_test=gt_2022[100000:]
    # set positives to label 1, Swap 0s with 1s
    Ytrain = 1 - gt_train[:]
    Ytest = 1 - gt_test[:]  


    print(f" Train data description : {pd.Categorical(Ytrain).describe()}")
    # Y_train = tf.keras.utils.to_categorical(Ytrain, 2)
    # Ytrain_tensor = torch.tensor(Ytrain, dtype=torch.long)  # Convert to tensor with dtype long
    # Y_train = torch.nn.functional.one_hot(Ytrain_tensor, num_classes=2).float()
    # print(Y_train)


    print(f" Test data description {pd.Categorical(Ytest).describe()}")
    # Y_test = tf.keras.utils.to_categorical(Ytest, 2)
    # Ytest_tensor = torch.tensor(Ytest, dtype=torch.long)  # Convert to tensor with dtype long
    # Y_test = torch.nn.functional.one_hot(Ytest_tensor, num_classes=2).float()
    # print(Y_test)
    return train, test, Ytrain, Ytest