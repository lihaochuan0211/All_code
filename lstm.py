import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import random
# 固定随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# 设置 Matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为SimHei
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题

def data_preprocess(features, window_size,step):
    data_X = []
    data_y = []
    for i in range(0, len(features) - window_size, step):
        data_X.append(features[i:i + window_size, :])
        data_y.append(features[i + window_size, 0])
    return np.array(data_X), np.array(data_y)


def plot_results(pred, real):

    relative_error = np.abs((real - pred) / real)
    mean_relative_error = np.mean(relative_error)
    relative_error_percentage = relative_error * 100
    num_low_error_points = np.sum(relative_error_percentage < 20)
    mse = mean_squared_error(real, pred)
    mae = np.mean(np.abs(real - pred))  # 计算MAE
    rmse = np.sqrt(mse)
    r2 = r2_score(real, pred)

    print("--------------------------------------------------------")
    print(f"测试集平均相对误差: {mean_relative_error:.6f}")
    print(f"相对误差低于20%的数据点数量: {num_low_error_points}")
    print(f"mse: {mse:.6f}, rmse: {rmse:.6f}, r2: {r2:.6f},mae: {mae:.6f}, mape:{mean_relative_error:.6f}")
    # 将评估指标保存到文本文件中
    eval_file_path = os.path.join(model_save_dir, f'{province}_lstm_{mean_relative_error:.2f}.txt')
    with open(eval_file_path, 'w') as f:
        f.write(f"测试集平均相对误差: {mean_relative_error:.6f}\n")
        f.write(f"相对误差低于20%的数据点数量: {num_low_error_points}\n")
        f.write(f"均方误差 (mse): {mse:.6f}, 均方根误差 (rmse): {rmse:.6f}, 决定系数 (r2): {r2:.6f},平均绝对误差 (mae): {mae:.6f}, 平均相对误差 (mape): {mean_relative_error:.6f}\n")


    model_save_path = os.path.join(model_save_dir, f'{province}_lstm_{mean_relative_error:.2f}.pth')
    torch.save(model.state_dict(), model_save_path)
    # print(f"Model saved as {model_save_path}")

    fig, ax1 = plt.subplots(figsize=(18, 9))
    months = pd.date_range(start='2019-03', periods=len(pred), freq='M').strftime('%Y-%m').tolist()

    ax1.plot(months, pred, label='pred', color='green')  # [i for i in range(len(pred))] month 的位置
    ax1.plot(months, real, label='real', color='blue')
    ax1.set_xlabel('年份', fontsize=15)
    ax1.set_ylabel('发电量', fontsize=15)
    ax1.legend(loc='upper left')
    ax2 = ax1.twinx()
    ax2.plot(months, relative_error_percentage, label='Relative Error (%)', color='red')  # same as above
    ax2.axhline(y=20, color='gray', linestyle='--', label='20% Error')
    ax2.set_ylabel('相对误差 (%)', fontsize=15)
    ax2.legend(loc='upper right')
    ax2.set_ylim(0, 100)

    ax1.tick_params(axis='x', labelsize=12)  # Change x-axis label size
    ax1.tick_params(axis='y', labelsize=15)  # Change y-axis label size
    ax2.tick_params(axis='y', labelsize=15)  # Change y-axis label size for the second axis (ax2)

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(model_save_dir, f'{province}_lstm_{mean_relative_error:.2f}.png'))
    plt.show()


def plot_all(pred, real):

    relative_error = np.abs((real - pred) / real)
    mean_relative_error_all = np.mean(relative_error)
    relative_error_percentage = relative_error * 100
    num_low_error_points = np.sum(relative_error_percentage < 20)
    mse = mean_squared_error(real, pred)
    mae = np.mean(np.abs(real - pred))  # 计算MAE
    rmse = np.sqrt(mse)
    r2 = r2_score(real, pred)

    print("--------------------------------------------------------")
    print(f"全数据集平均相对误差: {mean_relative_error_all:.6f}")
    print(f"全数据集相对误差低于20%的数据点数量: {num_low_error_points}")
    print(f"mse: {mse:.6f}, rmse: {rmse:.6f}, r2: {r2:.6f},mae: {mae:.6f}, mape:{mean_relative_error_all:.6f}")
    print("--------------------------------------------------------")
    # 将评估指标保存到文本文件中
    eval_file_path = os.path.join(model_save_dir, f'{province}_lstm_{test_mape:.2f}_all.txt')
    with open(eval_file_path, 'w') as f:
        f.write(f"全数据集平均相对误差: {mean_relative_error_all:.6f}\n")
        f.write(f"全数据集相对误差低于20%的数据点数量: {num_low_error_points}\n")
        f.write(f"均方误差 (mse): {mse:.6f}, 均方根误差 (rmse): {rmse:.6f}, 决定系数 (r2): {r2:.6f},平均绝对误差 (mae): {mae:.6f}, 平均相对误差 (mape): {mean_relative_error_all:.6f}\n")

    # 数据集划分边界
    N = len(pred)
    train_boundary = int(0.8 * N)
    val_boundary = int(0.9 * N)

    fig, ax1 = plt.subplots(figsize=(18, 9))
    ax1.plot([i for i in range(len(pred))], pred, label='预测', color='green')
    ax1.plot([i for i in range(len(pred))], real, label='实际', color='blue')
    ax1.set_xlabel('年份', fontsize=15)
    ax1.set_ylabel('发电量', fontsize=15)    
    # 添加分隔线
    ax1.axvline(x=train_boundary, color='gray', linestyle='--')
    ax1.axvline(x=val_boundary, color='gray', linestyle='--')
    
    ax1.legend(loc='upper left')
    ax2 = ax1.twinx()
    ax2.plot([i for i in range(len(pred))], relative_error_percentage, label='相对误差 (%)', color='red')
    ax2.axhline(y=20, color='gray', linestyle='--')
    ax2.set_ylabel('相对误差 (%)', fontsize=15)
    ax2.legend(loc='upper right')

    ax2.set_ylim(0, 100)

    ax1.tick_params(axis='x', labelsize=12)
    ax1.tick_params(axis='y', labelsize=15)
    ax2.tick_params(axis='y', labelsize=15)

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(model_save_dir, f'{province}_lstm_{test_mape:.2f}_all.png'))
    plt.show()


class ElectricDataset(Dataset):
    def __init__(self, data_X, data_y):
        super().__init__()
        self.data_X = data_X
        self.data_y = data_y

    def __len__(self):
        return len(self.data_X)

    def __getitem__(self, index):
        return torch.tensor(self.data_X[index], dtype=torch.float), torch.tensor(self.data_y[index], dtype=torch.float)


class ElectricLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(ElectricLSTM, self).__init__()
        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :]).squeeze()
        return out


if __name__ == '__main__':
    province = '甘肃省'
    project_dir = r'E:\Project\发电量预测'
    model_save_dir = r'E:\Project\发电量预测\Code\Model_result'

    df = pd.read_excel(os.path.join(project_dir, 'Code', '各省发电量_三次样条插值.xlsx'))
    electric = df[province].values[:216]
    train_df = pd.read_excel(os.path.join(project_dir, 'province_data', f'{province}_data.xlsx'))
    pr = train_df['降水量'].to_numpy()[:216]

    # plt.plot([i for i in range(len(electric))], electric)
    # plt.show()

    scaler_electric = MinMaxScaler()
    scaler_pr = MinMaxScaler()

    electric_scaled = scaler_electric.fit_transform(electric.reshape(-1, 1))
    pr_scaled = scaler_pr.fit_transform(pr.reshape(-1, 1))

    features = np.hstack([electric_scaled, pr_scaled])
    window_size = 3
    step = 1

    input_size = 2
    output_size = 1

    hidden_size = 60
    num_layers = 6
    batch_size = 85
    lr = 0.001
    epochs = 3225
    patience = 3225

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    data_X, data_y = data_preprocess(features, window_size, step)
    # 8:1:1
    X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.1, shuffle=False)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=1/9, shuffle=False)

    train_dataset = ElectricDataset(X_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    valid_dataset = ElectricDataset(X_valid, y_valid)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = ElectricDataset(X_test, y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    all_dataset = ElectricDataset(data_X, data_y)
    all_dataloader = DataLoader(all_dataset, batch_size=1, shuffle=False)

    model = ElectricLSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size)
    model.to(device)
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

        print(f"Epoch: {epoch + 1}, Train Loss: {avg_train_loss:.6f}, Valid Loss: {avg_valid_loss:.6f}")

        if avg_valid_loss < minimum_loss:
            minimum_loss = avg_valid_loss
        else:
            count += 1

        if count >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

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
    test_mape = np.mean(relative_error)   # 为了命名 把all的名字改成test_mape

    plot_results(test_pred, test_real)

    # Plot all data
    model.eval()
    all_pred = []
    all_real = []
    with torch.no_grad():
        for data_input, data_target in all_dataloader:
            data_input, data_target = data_input.to(device), data_target.to(device)
            out = model(data_input)
            all_pred.append(out.cpu().item())  # 如果验证集的batch_size不为1，那么就要用.numpy()，否则会报错
            all_real.append(data_target.cpu().item())

    # Convert lists to numpy arrays
    all_pred = np.array(all_pred).reshape(-1, 1)
    all_real = np.array(all_real).reshape(-1, 1)

    # Use inverse_transform method of the scaler
    all_pred = scaler_electric.inverse_transform(all_pred)
    all_real = scaler_electric.inverse_transform(all_real)

    plot_all(all_pred, all_real)