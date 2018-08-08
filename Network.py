import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

def scaleColumns(df):
    df = scaler.fit_transform(df)
    return df


train_file = os.getcwd() + '/train/all_features.csv'
test_file = os.getcwd() + '/test/all_features.csv'
trainval_data = pd.read_csv(train_file, sep=' ')
test_data = pd.read_csv(test_file, sep=' ')
scaler = preprocessing.MinMaxScaler()
trainval_features = trainval_data.iloc[:,4:]
trainval_label = trainval_data.iloc[:,0]
test_features = test_data.iloc[:,4:]
combinedFeature = trainval_features.append(test_features, ignore_index=True)

scombinedFeature = pd.DataFrame(scaleColumns(combinedFeature))
trainval_features = scombinedFeature.iloc[:trainval_features.shape[0],:]
test_features = scombinedFeature.iloc[trainval_features.shape[0]:,:]
train_id, val_id = \
    train_test_split(range(trainval_features.shape[0]), test_size=0.1, random_state=1)
train_features = trainval_features.iloc[train_id,:]
val_features = trainval_features.iloc[val_id,:]
train_label = trainval_label.iloc[train_id]
val_label = trainval_label.iloc[val_id]

class customDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data1, data2):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data1 = data1
        self.data2 = data2

    def __len__(self):
        return len(self.data1)

    def __getitem__(self, idx):
        sample1 = self.data1.iloc[idx, :].as_matrix()
        sample1 = sample1.astype('float')
        sample2 = self.data2.iloc[idx]
        sample2 = sample2.astype('float')
        return sample1, sample2

train_loader = DataLoader(customDataset(train_features, train_label), batch_size=128,
                        shuffle=True, num_workers=0)
val_loader = DataLoader(customDataset(val_features, val_label), batch_size=32,
                        shuffle=False, num_workers=0)
layers = []
layers.append(nn.Linear(train_features.shape[1], 80))
layers.append(nn.ReLU())
layers.append(nn.Linear(80, 50))
layers.append(nn.ReLU())
layers.append(nn.Linear(50, 10))
layers.append(nn.ReLU())
layers.append(nn.Linear(10,1))
Net = nn.Sequential(*layers)
Net = Net.cuda()
criteria = nn.BCEWithLogitsLoss()
step_loss_train = []
step_loss_val = []
acc_val = []
prec_val = []

best_accuracy = 0.
savepath = os.getcwd() + '/models'
if not os.path.isdir(savepath):
    os.makedirs(savepath)
else:
    fname = os.listdir(savepath)
    Net = torch.load(savepath + '/' + fname[-1])
    Net = Net.cuda()
optimizer = torch.optim.SGD(Net.parameters(), lr=0.001, momentum=0.9, nesterov=True, weight_decay=0.001)

for epoch in range(100):
    Net.train()
    for idx, (data, targets) in enumerate(train_loader):
        data = data.float()
        targets = targets.float()
        data = data.cuda()
        targets = targets.cuda()
        with torch.no_grad():
            data, targets = Variable(data), Variable(targets)
        optimizer.zero_grad()
        output = Net(data)
        loss = criteria(output.squeeze(1), targets)
        loss.backward()
        optimizer.step()
        step_loss_train.append(loss.item())
        # print('Training_loss: Epoch: {}, Iter: {}, Loss: {}'.format(epoch, idx, loss.item()))
    if epoch % 10 == 0:
        targetall = []
        outputall = []
        epoch_loss_val = []
        for idy, (valdata, valtargets) in enumerate(val_loader):
            valdata = valdata.float()
            valtargets = valtargets.float()
            valdata = valdata.cuda()
            valtargets = valtargets.cuda()
            with torch.no_grad():
                valdata, valtargets = Variable(valdata), Variable(valtargets)
            valout = Net(valdata)
            valoutsig = F.sigmoid(valout).squeeze(1)
            valloss = criteria(valout.squeeze(1), valtargets)
            epoch_loss_val.append(valloss.item())
            # print('Validation_loss: Epoch: {}, Iter: {}, Loss: {}'.format(epoch, idy, valloss.item()))
            outputall.extend(valoutsig.cpu().detach().numpy())
            targetall.extend(valtargets.cpu().detach().numpy())
        step_loss_val.append(np.mean(epoch_loss_val))
        outputall = [0 if i < 0.5 else 1 for i in np.array(outputall)]
        tn, fp, fn, tp = confusion_matrix(np.array(targetall), np.array(outputall)).ravel()
        auc = roc_auc_score(np.array(targetall), np.array(outputall))
        accur = (tp + tn)/float(len(targetall))
        precs = tp/(tp+fp)
        recall = tp/(tp + fn)
        f1score = 2*(precs*recall)/(precs + recall)

        print('Epoch: {} - Accuracy: {} - Precision: {} - Recall: {} - F1:{} - AUC: {}'.format(epoch, accur, precs, recall, f1score, auc))
        acc_val.append(accur)
        prec_val.append(precs)
        if f1score > best_accuracy:
            torch.save(Net, savepath + '/bestmodel' + str(epoch) + '.net')
            best_accuracy = f1score

fname = os.listdir(savepath)
newNet = torch.load(savepath + '/' + fname[0])
newNet = newNet.cuda()
test_features = Variable(torch.from_numpy(test_features.as_matrix().astype('float')).float().cuda())
testout = F.sigmoid(newNet(test_features))
np.savetxt(savepath + '/probablities.txt', testout.cpu().detach().numpy())


