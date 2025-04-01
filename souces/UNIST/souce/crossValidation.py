import collections
import glob
import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import tqdm
import matplotlib.pyplot as plt

# デバイスの設定
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_data(inputX_dir, inputY_dir):
    X = []
    forX = {}
    y = []

    #get patient and spot list
    print("Get patient and spot list")
    lists = glob.glob(inputX_dir + "/*_*.features.pt")
    patient = collections.defaultdict(list)
    with tqdm.tqdm(total=len(lists)) as pbar:
        for (p, s) in map(lambda x: x.split("/")[-1][:-12].split("_"), lists):
            patient[p].append(s)
            pbar.update()

    #load x data(pt)
    print("Get image features data")
    spots = {}
    with tqdm.tqdm(total=sum(map(len, patient.values()))) as pbar:
        for p in patient:
            spots[p] = {}
            forX[p] = {}
            for s in patient[p]:
                spots[p][s] = []
                forX[p][s] = {}
                dataX = torch.load(inputX_dir + "/" + p + "_" + s + ".features.pt", weights_only=True)
                sorted_dataX = {k: v for k, v in sorted(dataX.items(), key=lambda item: item[0])}
                for key, value in sorted_dataX.items():
                    spots[p][s].append(key)
                    forX[p][s][key] = value.squeeze()
                pbar.update()
    
    #load y data(npz)
    print("Get counts data")
    new_spots = {}
    with tqdm.tqdm(total=sum(map(len, patient.values()))) as pbar:
        for p in patient:
            new_spots[p] = {}
            for s in patient[p]:
                new_spots[p][s] = []
                with open(inputY_dir + "/" + p + "_" + s + ".counts.pkl", "rb") as f:
                    datay = pickle.load(f)
                filtered_datay = {}
                for patch_rf in spots[p][s]:
                    patch = patch_rf.split("_")[0]
                    if patch in datay:
                        filtered_datay[patch_rf] = datay[patch]
                        new_spots[p][s].append(patch_rf)
                for _, value in filtered_datay.items():
                    y.append(torch.from_numpy(value))
                pbar.update()
    y = torch.stack(y)
    spots = new_spots
    
    print("reGet image features data")
    with tqdm.tqdm(total=sum(map(len, spots.values()))) as pbar:
        for p in spots:
            for s in spots[p]:
                for i in spots[p][s]:
                    X.append(forX[p][s][i])
                pbar.update()
    X = torch.stack(X)

    if len(X) != len(y):
        print("Error: X and y have different lengths")
        return None, None
    
    return X, y, spots

def prepare_data(y):
    #select top 250 columns
    mean_values = torch.mean(y, dim=0)
    _, top_indices = torch.topk(mean_values, 250)
    y = y[:, top_indices]

    #normalize
    Z = y.sum(dim=1, keepdim=True)
    y = torch.log((1 + y) / (Z + y.shape[1]))
    return y

def create_dataloader(X, y, batch_size):
    dataset = TensorDataset(X, y)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader

class LinearModel(nn.Module):
    """
    def __init__(self, input_dim, output_dim):
        super(LinearModel, self).__init__()

        self.fc1 = nn.Linear(input_dim, int((input_dim + output_dim) / 2)) 
        self.fc2 = nn.Linear(int((input_dim + output_dim) / 2), output_dim) 
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    """
    def __init__(self, input_dim, output_dim):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)

class ImprovedMLPModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ImprovedMLPModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

class BinaryModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BinaryModel, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(inplace=True),
            #nn.Dropout(0.5),
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(inplace=True),
            #nn.Dropout(0.5),
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(inplace=True),
            #nn.Dropout(0.5),
            nn.Linear(input_dim, output_dim),
            #nn.BatchNorm1d(output_dim),
            #nn.ReLU(inplace=True),
            nn.Linear(output_dim, output_dim),
        )
    
    def forward(self, x):
        x = self.classifier(x)
        return x
    
def train_model(model, data_loader, criterion, optimizer, num_epochs, batch_size):
    for epoch in range(num_epochs):
        model.train()
        for batch_X, batch_y in data_loader:
            if batch_X.size(0) == 1:
                continue
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            # 順伝播
            outputs = model(batch_X)
            loss = torch.sum((outputs - batch_y) ** 2) / batch_y.size(1)
            #loss = criterion(outputs, batch_y)
            
            # 逆伝播と最適化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # GPUメモリの開放
            del batch_X, batch_y, outputs
            torch.cuda.empty_cache()
        
        loss = loss.item() / batch_size
        #loss_list.append(loss.item())
        #print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.4f}')
    
    # GPUメモリの開放
    del loss
    torch.cuda.empty_cache()
    print("学習完了")

def cross_validation(inputX_dir, inputY_dir, input_dim, output_dim, batch_size, num_epochs, learning_rate, momentum):
    # データの読み込み
    X, y, spots = load_data(inputX_dir, inputY_dir)
    y = prepare_data(y).to(device)
    X = X.to(device)
    X = X.to(torch.float64)
    y = y.to(torch.float64)

    print(X.size())
    print(y.size())

    results = []

    for i, test_patient in enumerate(spots.keys()):
        print(f"Cross-validation for patient: {test_patient}({i+1}/23)")

        # 訓練データとテストデータに分割
        train_indices = []
        test_indices = []
        i = 0
        for p in spots.keys():
            for v in spots[p].keys():
                for s in spots[p][v]:
                    if p == test_patient:
                        test_indices.append(i)
                    else:
                        train_indices.append(i)
                    i += 1

        X_train, y_train = X[train_indices], y[train_indices]
        X_test, y_test = X[test_indices], y[test_indices]

        # データローダーの作成
        train_loader = create_dataloader(X_train, y_train, batch_size)

        # モデルのインスタンス化
        #model = LinearModel(input_dim, output_dim).to(device)
        #model = ImprovedMLPModel(input_dim, 512, output_dim).to(device)
        model = BinaryModel(input_dim, output_dim).to(device)
        model = model.to(torch.float64)

        # 損失関数とオプティマイザの定義
        #criterion = nn.MSELoss()
        criterion = nn.SmoothL1Loss()  # Huber損失
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

        # モデルの学習
        train_model(model, train_loader, criterion, optimizer, num_epochs, batch_size)

        # テストデータで予測
        model.eval()
        with torch.no_grad():
            y_pred = model(X_test).cpu()

        results.append((test_patient, y_pred.cpu(), y_test.cpu()))

        # GPUメモリの開放
        del model, criterion, optimizer, X_train, y_train, X_test, y_test, train_loader, y_pred
        torch.cuda.empty_cache()

    return results

def main():
    inputX_dir = "/host/UNIST/data/forTraining/imageFeatures"
    inputY_dir = "/host/STCLIP/data/forTraining/countsData"
    output_dir = "/host/UNIST/output/cross"  # pthファイルの保存先ディレクトリ
    input_dim = 1536
    output_dim = 250
    batch_size = 128
    num_epochs = 200
    learning_rate = 1e-4
    momentum = 0.8

    # ディレクトリが存在しない場合は作成
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # クロスバリデーションの実行
    results = cross_validation(inputX_dir, inputY_dir, input_dim, output_dim, batch_size, num_epochs, learning_rate, momentum)

    # 結果の保存
    for (patient, y_pred, y_test) in (results):
        torch.save((patient, y_pred, y_test), os.path.join(output_dir, f'result_{patient}.pt'))
        print(f"結果が {os.path.join(output_dir, f'result_{patient}.pt')} に保存されました")
        print(y_pred.size())
        print(y_test.size())

if __name__ == "__main__":
    main()