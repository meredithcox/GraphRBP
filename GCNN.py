import torch
from torch_geometric.data import Data
import pickle
import one_hot
from torch_geometric.data import InMemoryDataset
from torch_geometric.nn import GraphConv, TopKPooling
from torch_geometric.nn import GraphConv, TopKPooling, GatedGraphConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
import numpy as np

class RBPDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(RBPDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['pos_sequences.txt']

    def download(self):
        pass

    def load_data(self):

        f = open('pos_ei_FXR1.pkl', 'rb')
        ei_pos = pickle.load(f)

        f = open('neg_ei_FXR1.pkl', 'rb')
        ei_neg = pickle.load(f)

        s_pos_txt = open('pos_sequences_FXR1.txt', 'r')
        s_pos = []
        for line in s_pos_txt:
            s_pos.append([one_hot.to_one_hot(line.rstrip())])
        s_neg_txt = open('neg_sequences_FXR1.txt', 'r')
        s_neg = []
        for line in s_neg_txt:
            s_neg.append([one_hot.to_one_hot(line.rstrip())])
        return ei_pos, ei_neg, s_pos, s_neg

    def process(self):
        ei_pos, ei_neg, s_pos, s_neg = self.load_data()
        data_list = []
        for i in range(12000):
            data = Data(edge_index=torch.Tensor(ei_pos[i]), x=torch.Tensor(s_pos[i]).squeeze().t(), y=torch.Tensor([1]))
            data_list.append(data)
            data = Data(edge_index=torch.Tensor(ei_neg[i]), x=torch.Tensor(s_neg[i]).squeeze().t(), y=torch.Tensor([0]))
            data_list.append(data)
            if i % 1000 == 0:
                print(i)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

dataset = RBPDataset(root='../')
#dataset = dataset.shuffle()
train_dataset = dataset[:20000]
val_dataset = dataset[20000:22000]
test_dataset = dataset[22000:25000]
k_fold = dataset[:10000]
print(len(train_dataset), len(val_dataset), len(test_dataset))

batch_size= 500
train_loader = DataLoader(train_dataset, batch_size=batch_size)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)
embed_dim = 256


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = GraphConv(4, embed_dim)
        self.pool1 = TopKPooling(embed_dim, ratio=0.8)
        self.conv2 = GraphConv(embed_dim, embed_dim)
        self.pool2 = TopKPooling(embed_dim, ratio=0.8)
        self.conv3 = GraphConv(embed_dim, embed_dim)
        self.pool3 = TopKPooling(embed_dim, ratio=0.8)
        self.item_embedding = torch.nn.Embedding(2, embedding_dim=embed_dim)
        self.lin1 = torch.nn.Linear(embed_dim*2, embed_dim)
        self.lin2 = torch.nn.Linear(embed_dim, int(embed_dim/2))
        self.lin3 = torch.nn.Linear(int(embed_dim/2), 1)
        self.bn1 = torch.nn.BatchNorm1d(embed_dim)
        self.bn2 = torch.nn.BatchNorm1d(int(embed_dim/2))
        self.act1 = torch.nn.ReLU()
        self.act2 = torch.nn.ReLU()

    def forward(self, data):
        x, edge_index, batch = data.x.float(), data.edge_index.float(), data.batch.float()

        x = F.relu(self.conv1(x.float(), edge_index.long()))

        x1 = torch.cat([gmp(x.long(), batch.long()), gap(x.long(), batch.long())], dim=1)

        x = F.relu(self.conv2(x.float(), edge_index.long()))

        x2 = torch.cat([gmp(x.long(), batch.long()), gap(x.long(), batch.long())], dim=1)

        x = F.relu(self.conv3(x.float(), edge_index.long()))

        x, edge_index, _, batch, _,_ = self.pool3(x.long(), edge_index.long(), None, batch.long())
        x3 = torch.cat([gmp(x.long(), batch.long()), gap(x.long(), batch.long())], dim=1)

        x = x1 + x2 + x3

        x = self.lin1(x)
        x = self.act1(x)
        x = self.lin2(x)
        x = self.act2(x)
        x = F.dropout(x, p=0.2, training=self.training)

        x = torch.sigmoid(self.lin3(x)).squeeze(1)

        return x

model = Net()
optimizer = torch.optim.Adam(model.parameters(), lr=.001)
crit = torch.nn.BCELoss()

def train():
    model.train()

    torch.manual_seed(0)
    loss_all = 0
    for data in train_loader:
        optimizer.zero_grad()
        output = model(data)
        label = data.y
        loss = crit(output, label)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    return loss_all / len(train_dataset)


def evaluate(loader):
    model.eval()

    predictions = []
    labels = []

    with torch.no_grad():
        for data in loader:
            pred = model(data).numpy()

            label = data.y.numpy()
            predictions.append(pred)
            labels.append(label)

    predictions = np.hstack(predictions)
    labels = np.hstack(labels)

    acc = roc_auc_score(labels, predictions)
    #if acc > .91:
        #torch.save(model.state_dict(), 'gcnn_model_91.pt')

    return acc

'''
patience = 10
patience_counter = patience
f = open('train_test_accs.txt','w')
best_val_acc = 0
for epoch in range(100):
    loss = train()
    train_acc = evaluate(train_loader)
    val_acc = evaluate(val_loader)
    test_acc = evaluate(test_loader)
    f.write(str(train_acc) + "\t" + str(test_acc))
    print('Epoch: {:03d}, Loss: {:.5f}, Train Auc: {:.5f}, Val Auc: {:.5f}, Test Auc: {:.5f}'.
          format(epoch, loss, train_acc, val_acc, test_acc))
    if val_acc > best_val_acc:
        torch.save(model.state_dict(), 'checkpoint.pt')
        best_val_acc = val_acc
        patience_counter = patience
    else:
        patience_counter -= 1
        if patience_counter <= 0:
            model.load_state_dict(torch.load('checkpoint.pt'))  # recover the best model so far
            break

'''

patience = 10
patience_counter = patience
scores = []
best_svr = SVR(kernel='rbf')
cv = KFold(n_splits=10, shuffle=False)
for train_index, test_index in cv.split(dataset[:10000]):
    model = Net()
    optimizer = torch.optim.Adam(model.parameters(), lr=.0005)
    crit = torch.nn.BCELoss()
    print(type(dataset))
    train_loader = DataLoader([dataset[int(i)] for i in train_index], batch_size=batch_size)
    test_loader = DataLoader([dataset[int(i)] for i in test_index], batch_size=batch_size)
    best_test_acc = 0
    for epoch in range(30):
        loss = train()
        train_acc = evaluate(train_loader)
        test_acc = evaluate(test_loader)
        print('Epoch: {:03d}, Loss: {:.5f}, Train Auc: {:.5f}, Test Auc: {:.5f}'.
              format(epoch, loss, train_acc, test_acc))
        if test_acc > best_test_acc:
            torch.save(model.state_dict(), 'checkpoint.pt')
            best_test_acc = test_acc
            patience_counter = patience
        else:
            patience_counter -= 1
            if patience_counter <= 0:
                model.load_state_dict(torch.load('checkpoint.pt'))  # recover the best model so far
                break
    scores.append(best_test_acc)
