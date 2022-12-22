import torch 
import scipy
from scipy.stats import spearmanr
import numpy as np


def Scoring(df_te, df_predicted):
    
    """
    Calculate different scores
    Input: prediction and real labels
    Output: Pearon's correlation coefficient, RMSE and AUC
    """
    df = {
    "true": df_te['tm'],
    "predicted": df_predicted['tm']
}
    pearson = df.corr(method='pearson')
    rmse = mean_squared_error(df_te['tm'], df_predicted['tm'], squared=False)
    auc = metrics.roc_auc_score(df_te['tm'], df_predicted['tm'])
    
    print('Pearson: %.3f, RMSE %.3f, AUC: %.3f' %(pearson, rmse, auc))
    return pearson, rmse, auc


def train_epoch(model, optimizer, criterion, train_loader, epoch):
    """
    Function used to train the model
    Input: Model,
    Optimizer (Adam in the notebooks),
    criterion (MSE loss in notebook), 
    train_loader : created with pytorch dataloader, from a dataset created with EnzymeDataset
    epoch: number of epoch 
    
    Output : loss and Spearman's coefficient
    """
    model.train()
    rho = 0 
    train_loss = 0 
    for batch_idx, (seq, target,num) in enumerate(train_loader):
        if torch.cuda.is_available():
            seq = seq.cuda()
            target = target.cuda()
            num = num.cuda()
        optimizer.zero_grad()
        
        output = model(seq,num)
        loss = criterion(output.squeeze(), target)
        train_loss += loss.item()
        loss.backward()
        
        optimizer.step()
        # calculate Spearman's rank correlation coefficient
        p, _ = scipy.stats.spearmanr(target.cpu().detach().numpy(), output.squeeze().cpu().detach().numpy())
        rho += p
    
    train_loss /= len(train_loader)
    
    if epoch % 50 == 0 :
        print(   f"Train Epoch: {epoch} " f" loss={train_loss:0.2e} " )

    rho = rho / len(train_loader)
    return train_loss , rho


def test_epoch(model, criterion, test_loader):
    """
    Function used to test the model, with the test data
    Input: Model,
   
    criterion (MSE loss in notebook), 
    test_loader : created with pytorch dataloader, from a dataset created with EnzymeDataset
    
    
    Return: loss and Spearman's coefficient
    """

    model = model.eval()
    test_loss = 0
    rho = 0
    with torch.no_grad():
        for seq, target,num in test_loader:
            if torch.cuda.is_available():
                seq = seq.cuda()
                target = target.cuda()
                num = num.cuda()
            output = model(seq,num)
            test_loss += criterion(output.squeeze(), target).item()  # sum up batch loss
            # calculate pearson correlation 
            #pearson, rmse, auc = Scoring(target.cpu().detach(), output.cpu().detach())
            p, _ =  scipy.stats.spearmanr(target.cpu().detach().numpy(), output.cpu().detach().numpy())
            rho += p
            

    test_loss /= len(test_loader)
    rho = rho / len(test_loader)
    print(
        f"Test set: Average loss: {test_loss:0.2e} "
    )

    return test_loss ,rho



def split_train_test(df,frac,seed=24,verbose=True):
    """split into train and test sets
    Args: 
    df : pandas dataframe
    frac : scalar
    verbose : boolean , print infos
    
    Returns
    train_df , val_df : pandas dataframes
    """
    train_df = df.sample(frac=0.8,random_state=24)
    val_df = df.drop(train_df.index)

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    if verbose: 
        print('train_df has shape : {} \n test_df has shape :  {}'.format(train_df.shape,val_df.shape))

    return train_df , val_df



def predict(model,test_loader):
    """infer model predictions for submission
    Args: 
    model : pyotrch model
    test_loader : pytorch data loader
    
    Returns
    output : numpy array, predictions
    """
    model = model.eval()
    preds = []
    with torch.no_grad():
        for seq, target,num in test_loader:
            if torch.cuda.is_available():
                seq = seq.cuda()
                target = target.cuda()
                num = num.cuda()
            output = model(seq,num)
            preds.append(output.to('cpu').numpy())
            
    output = np.concatenate(preds)
    return output

def encode_seq(sequence, max_length):
    """
    Encode the amino acid sequence with one-hot-encoding
    Input: sequence and max length of the sequence
    Output:  encoded sequence
    """
    alphabet = ['A', 'C', 'D', 'E', 'F', 'G','H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'] # aa letters
    char_to_int = dict((c, i) for i, c in enumerate(alphabet)) 
    integer_encoded = [char_to_int[char] for char in sequence] #each character becomes int
    onehot_encoded = list()
    for value in integer_encoded:
        letter = [0 for _ in range(len(alphabet))] #0 for all letters
        letter[value] = 1 #modify the column corresponding to the letter to 1
        onehot_encoded.append(letter) #put in the array (1 letter = 1 array of 20 columns)
    
    ar =   np.transpose(np.array(onehot_encoded)) #Transpose to have the right shape for the CNN
    zeros = np.zeros([len(alphabet),max_length - len(integer_encoded)] )
    onehot_encoded = np.concatenate((ar, zeros), axis = 1) #zero padding


    return onehot_encoded #we have all arrays, corresponding to the whole sequence

