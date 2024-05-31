import os
import random
import numpy as np
from bayes_opt import BayesianOptimization, UtilityFunction
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd  # 导入csv文件的库
import warnings  # 避免一些可以忽略的报错
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime


# 固定随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)


class ElectricDataset(Dataset):
    def __init__(self, data_X, data_y):
        super().__init__()
        self.data_X = data_X
        self.data_y = data_y

    def __len__(self):
        return len(self.data_X)

    def __getitem__(self, index):
        return torch.tensor(self.data_X[index], dtype=torch.float), torch.tensor(self.data_y[index], dtype=torch.float)


class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers, output_dim, dim_feedforward,dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        #self.input_dim = input_dim
        #self.embed_dim = embed_dim
        #self.output_dim = output_dim
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.positional_encoding = nn.Parameter(self._generate_positional_encoding(embed_dim, 5000))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(embed_dim, output_dim)

    def _generate_positional_encoding(self, embed_dim, max_len):
        pos = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(torch.log(torch.tensor(10000.0)) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        pe = pe.unsqueeze(0)
        return pe

    def forward(self, x):
        x = self.embedding(x)
        x = x + self.positional_encoding[:, :x.size(1), :]
        x = self.transformer_encoder(x)
        x = self.decoder(x[:, -1, :]).squeeze()
        return x


def data_preprocess(features, window_size, step):
    data_X = []
    data_y = []
    for i in range(0, len(features) - window_size, step):
        data_X.append(features[i:i + window_size, :])
        data_y.append(features[i + window_size, 0])
    return np.array(data_X), np.array(data_y)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def lstm_cv(dim_feedforward, num_layers, learning_rate, window_size, batch_size, epochs):
    # 调整超参数的数据类型
    window_size = int(window_size)
    dim_feedforward = int(dim_feedforward)
    num_layers = int(num_layers)
    epochs = int(epochs)
    batch_size = int(batch_size)

    lr = learning_rate

    input_dim = 2
    output_dim = 1

    embed_dim = 72   # 嵌入维度
    num_heads = 8    # 多头注意力机制的头数
    dropout = 0.1    # dropout
    patience = 150

    province = '云南省'
    project_dir = r'E:\Project\发电量预测'
    model_save_dir = r'E:\Project\发电量预测\Code\Model_result'

    df = pd.read_excel(os.path.join(project_dir, 'Code', '各省发电量_三次样条插值.xlsx'))
    electric = df[province].values[:216]
    train_df = pd.read_excel(os.path.join(project_dir, 'province_data', f'{province}_data.xlsx'))
    pr = train_df['降水量'].to_numpy()[:216]

    scaler_electric = MinMaxScaler()
    scaler_pr = MinMaxScaler()

    electric_scaled = scaler_electric.fit_transform(electric.reshape(-1, 1))
    pr_scaled = scaler_pr.fit_transform(pr.reshape(-1, 1))

    features = np.hstack([electric_scaled, pr_scaled])
    step = 1

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    data_X, data_y = data_preprocess(features, window_size, step)
    # 8:1:1
    X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.1, shuffle=False)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=1 / 9, shuffle=False)

    train_dataset = ElectricDataset(X_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    valid_dataset = ElectricDataset(X_valid, y_valid)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = ElectricDataset(X_test, y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    # 创建模型
    model = TimeSeriesTransformer(input_dim, embed_dim, num_heads, num_layers, output_dim, dim_feedforward,dropout).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    minimum_loss = 1e10
    count = 0

    for epoch in range(epochs):

        avg_train_losses = []
        model.train()
        for data_input, data_target in train_dataloader:
            data_input, data_target = data_input.to(device), data_target.to(device)
            optimizer.zero_grad()
            out = model(data_input)
            loss = criterion(out, data_target)
            loss.backward()
            optimizer.step()
            avg_train_losses.append(loss.cpu().item())

        model.eval()
        avg_valid_losses = []
        with torch.no_grad():
            for data_input, data_target in valid_dataloader:
                data_input, data_target = data_input.to(device), data_target.to(device)
                out = model(data_input)
                loss = criterion(out, data_target)
                avg_valid_losses.append(loss.cpu().item())

        avg_train_loss = sum(avg_train_losses) / len(avg_train_losses)
        avg_valid_loss = sum(avg_valid_losses) / len(avg_valid_losses)

        # print(f"Epoch: {epoch + 1}, Train Loss: {avg_train_loss:.6f}, Valid Loss: {avg_valid_loss:.6f}")

        if avg_valid_loss < minimum_loss:
            minimum_loss = avg_valid_loss
        else:
            count += 1

        if count >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

        def mse(pred_y, true_y):
            return np.mean((pred_y - true_y) ** 2)

    model.eval()
    test_pred = []
    test_real = []
    with torch.no_grad():
        for data_input, data_target in test_dataloader:
            data_input, data_target = data_input.to(device), data_target.to(device)
            out = model(data_input)
            test_pred.append(out.cpu().item())  # 如果验证集的batch_size不为1，那么就要用.numpy()，否则会报错
            test_real.append(data_target.cpu().item())

    # Convert lists to numpy arrays
    test_pred = np.array(test_pred).reshape(-1, 1)
    test_real = np.array(test_real).reshape(-1, 1)

    # Use inverse_transform method of the scaler
    test_pred = scaler_electric.inverse_transform(test_pred)
    test_real = scaler_electric.inverse_transform(test_real)

    relative_error = np.abs((test_real - test_pred) / test_real)
    test_mape = np.mean(relative_error)  # 为了命名 把all的名字改成test_mape

    return -test_mape


# 定义超参数的搜索范围
pbounds = {
    'window_size': (1, 20),
    'dim_feedforward': (10, 200),
    'num_layers': (1, 10),
    'learning_rate': (0.001, 0.01),
    'epochs': (100, 900),
    "batch_size": (32, 100),
    "dropout": (0.1, 0.5),
}

# 读取数据
optimizer = BayesianOptimization(f=lstm_cv, pbounds=pbounds, random_state=1)

# 创建一个UtilityFunction对象
uf = UtilityFunction(kind="ucb", kappa=3, xi=1)

# 使用自定义的UtilityFunction对象
optimizer._space._util = uf

# optimizer.maximize(init_points=5, n_iter=20)
start_time = datetime.now()
optimizer.maximize(init_points=5, n_iter=50)

print(optimizer.max)
