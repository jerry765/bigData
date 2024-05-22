import torch
from torch_geometric.datasets import WikipediaNetwork, Coauthor
import pickle
from torch_geometric.nn import GATConv
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from datetime import datetime
from torch_geometric.data import Data
import numpy as np

# 确保 ./data 目录存在
data_dir = './data'
os.makedirs(data_dir, exist_ok=True)

# 加载Squirrel数据集并保存在指定目录
squirrel_dataset = WikipediaNetwork(root=os.path.join(data_dir, 'Squirrel'), name='Squirrel')
squirrel_data = squirrel_dataset[0]

# 加载Coauthor CS数据集并保存在指定目录
coauthor_dataset = Coauthor(root=os.path.join(data_dir, 'CoauthorCS'), name='CS')
coauthor_data = coauthor_dataset[0]

# 保存数据集到 ./data 目录
with open(os.path.join(data_dir, 'squirrel_data.pkl'), 'wb') as f:
    pickle.dump(squirrel_data, f)

with open(os.path.join(data_dir, 'coauthor_data.pkl'), 'wb') as f:
    pickle.dump(coauthor_data, f)

# 从 ./data 目录加载数据集
with open(os.path.join(data_dir, 'squirrel_data.pkl'), 'rb') as f:
    loaded_squirrel_data = pickle.load(f)

with open(os.path.join(data_dir, 'coauthor_data.pkl'), 'rb') as f:
    loaded_coauthor_data = pickle.load(f)


# 为 CoauthorCS 数据集添加 train_mask, val_mask 和 test_mask
def add_masks(data, train_ratio=0.6, val_ratio=0.2):
    num_nodes = data.y.shape[0]
    indices = np.arange(num_nodes)
    np.random.shuffle(indices)

    train_size = int(train_ratio * num_nodes)
    val_size = int(val_ratio * num_nodes)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask


# 添加 CoauthorCS 数据集掩码
add_masks(loaded_coauthor_data)


# 修正 Squirrel 数据集的掩码
def select_first_task_mask(data):
    data.train_mask = data.train_mask[:, 0]
    data.val_mask = data.val_mask[:, 0]
    data.test_mask = data.test_mask[:, 0]


select_first_task_mask(loaded_squirrel_data)

# 验证加载的数据是否正确
print(loaded_squirrel_data)
print(loaded_coauthor_data)


# 定义 GAT 模型
class GAT(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, 8, heads=8, dropout=0.6)  # 第一层 GATConv，8 个头
        self.conv2 = GATConv(8 * 8, out_channels, heads=1, dropout=0.6)  # 第二层 GATConv，1 个头

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)  # Dropout 层
        x = F.elu(self.conv1(x, edge_index))  # 第一层卷积并应用 ELU 激活函数
        x = F.dropout(x, p=0.6, training=self.training)  # Dropout 层
        x = self.conv2(x, edge_index)  # 第二层卷积
        return F.log_softmax(x, dim=1)  # Log-Softmax 激活函数


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设置设备为 GPU 或 CPU

# 定义小批次训练的数据加载器
from torch_geometric.loader import DataLoader


def create_data_loader(data, batch_size=32):
    dataset = [data]  # Assuming each dataset is loaded as a single graph
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


coauthor_loader = create_data_loader(loaded_coauthor_data)
squirrel_loader = create_data_loader(loaded_squirrel_data)


# 训练函数
def train_batch(loader, model, optimizer):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)


# 测试函数
def test_batch(loader, model):
    model.eval()
    total_correct = 0
    total_examples = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=1)
            total_correct += int((pred[data.test_mask] == data.y[data.test_mask]).sum())
            total_examples += int(data.test_mask.sum())
    return total_correct / total_examples


# 准备绘图目录
plot_dir = './plots'
os.makedirs(plot_dir, exist_ok=True)

# 准备模型保存目录
model_dir = './models'
os.makedirs(model_dir, exist_ok=True)

# 定义时间戳，用于保存文件名
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# 存储损失和准确率
coauthor_losses = []
coauthor_accs = []
squirrel_losses = []
squirrel_accs = []

# 初始化模型和优化器
coauthor_model = GAT(loaded_coauthor_data.num_node_features, coauthor_dataset.num_classes).to(device)
squirrel_model = GAT(loaded_squirrel_data.num_node_features, squirrel_dataset.num_classes).to(device)

coauthor_optimizer = torch.optim.Adam(coauthor_model.parameters(), lr=0.005, weight_decay=5e-4)
squirrel_optimizer = torch.optim.Adam(squirrel_model.parameters(), lr=0.005, weight_decay=5e-4)

# 训练和测试循环
for epoch in range(200):
    coauthor_loss = train_batch(coauthor_loader, coauthor_model, coauthor_optimizer)
    squirrel_loss = train_batch(squirrel_loader, squirrel_model, squirrel_optimizer)

    coauthor_acc = test_batch(coauthor_loader, coauthor_model)
    squirrel_acc = test_batch(squirrel_loader, squirrel_model)

    coauthor_losses.append(coauthor_loss)
    coauthor_accs.append(coauthor_acc)
    squirrel_losses.append(squirrel_loss)
    squirrel_accs.append(squirrel_acc)

    print(f'Epoch {epoch + 1}, Coauthor Loss: {coauthor_loss:.4f}, Coauthor Acc: {coauthor_acc:.4f}')
    print(f'Epoch {epoch + 1}, Squirrel Loss: {squirrel_loss:.4f}, Squirrel Acc: {squirrel_acc:.4f}')

    # 保存模型
    coauthor_model_path = os.path.join(model_dir, f'coauthor_gat_model_epoch_{epoch + 1}_{timestamp}.pth')
    squirrel_model_path = os.path.join(model_dir, f'squirrel_gat_model_epoch_{epoch + 1}_{timestamp}.pth')
    torch.save(coauthor_model.state_dict(), coauthor_model_path)
    torch.save(squirrel_model.state_dict(), squirrel_model_path)

# 绘制并保存损失曲线和准确率曲线
coauthor_loss_plot_path = os.path.join(plot_dir, f'coauthor_loss_plot_{timestamp}.png')
squirrel_loss_plot_path = os.path.join(plot_dir, f'squirrel_loss_plot_{timestamp}.png')
coauthor_acc_plot_path = os.path.join(plot_dir, f'coauthor_acc_plot_{timestamp}.png')
squirrel_acc_plot_path = os.path.join(plot_dir, f'squirrel_acc_plot_{timestamp}.png')

# 绘制损失曲线
plt.figure()
plt.plot(range(1, 201), coauthor_losses, label='Coauthor Loss')
plt.plot(range(1, 201), squirrel_losses, label='Squirrel Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.savefig(coauthor_loss_plot_path)
plt.savefig(squirrel_loss_plot_path)
plt.close()

# 绘制准确率曲线
plt.figure()
plt.plot(range(1, 201), coauthor_accs, label='Coauthor Acc')
plt.plot(range(1, 201), squirrel_accs, label='Squirrel Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')
plt.legend()
plt.savefig(coauthor_acc_plot_path)
plt.savefig(squirrel_acc_plot_path)
plt.close()
