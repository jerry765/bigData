import torch
from torch_geometric.datasets import WikipediaNetwork, Planetoid
import pickle
from torch_geometric.nn import GATConv
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from datetime import datetime

# 确保 ./data 目录存在
data_dir = './data'
os.makedirs(data_dir, exist_ok=True)

# 加载Squirrel数据集并保存在指定目录
squirrel_dataset = WikipediaNetwork(root=os.path.join(data_dir, 'Squirrel'), name='Squirrel')
squirrel_data = squirrel_dataset[0]

# 加载Cora数据集并保存在指定目录
cora_dataset = Planetoid(root=os.path.join(data_dir, 'Cora'), name='Cora')
cora_data = cora_dataset[0]

# 保存数据集到 ./data 目录
with open(os.path.join(data_dir, 'squirrel_data.pkl'), 'wb') as f:
    pickle.dump(squirrel_data, f)

with open(os.path.join(data_dir, 'cora_data.pkl'), 'wb') as f:
    pickle.dump(cora_data, f)

# 从 ./data 目录加载数据集
with open(os.path.join(data_dir, 'cora_data.pkl'), 'rb') as f:
    loaded_cora_data = pickle.load(f)

# 验证加载的数据是否正确
print(loaded_cora_data)


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


def train(data, model, optimizer):
    model.train()  # 设置模型为训练模式
    optimizer.zero_grad()  # 清空梯度
    out = model(data.x.to(device), data.edge_index.to(device))  # 模型前向传播
    train_mask = data.train_mask  # 获取训练掩码
    loss = F.nll_loss(out[train_mask], data.y[train_mask].to(device))  # 计算损失
    loss.backward()  # 反向传播
    optimizer.step()  # 更新模型参数
    return loss.item()  # 返回损失值


def test(data, model):
    model.eval()  # 设置模型为评估模式
    logits, accs = model(data.x.to(device), data.edge_index.to(device)), []  # 模型前向传播
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):  # 遍历训练、验证和测试掩码
        pred = logits[mask].max(1)[1]  # 预测类别
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()  # 计算准确率
        accs.append(acc)  # 保存准确率
    return accs  # 返回准确率


# 准备绘图目录
plot_dir = './plots'
os.makedirs(plot_dir, exist_ok=True)

# 准备模型保存目录
model_dir = './models'
os.makedirs(model_dir, exist_ok=True)

# 定义时间戳，用于保存文件名
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# 存储损失和准确率
losses = []
train_accs = []
val_accs = []
test_accs = []

# 使用加载的数据集进行训练和测试
model = GAT(loaded_cora_data.num_node_features, cora_dataset.num_classes).to(device)  # 初始化模型
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)  # 初始化优化器

# 训练和测试
for epoch in range(200):
    loss = train(loaded_cora_data, model, optimizer)  # 训练模型
    train_acc, val_acc, test_acc = test(loaded_cora_data, model)  # 测试模型

    # 存储每个epoch的损失和准确率
    losses.append(loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    test_accs.append(test_acc)

    print(f'Epoch {epoch + 1}, Loss: {loss:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')

    # 保存模型
    model_path = os.path.join(model_dir, f'gat_model_epoch_{epoch + 1}_{timestamp}.pth')
    torch.save(model.state_dict(), model_path)

# 绘制并保存损失曲线和准确率曲线
loss_plot_path = os.path.join(plot_dir, f'loss_plot_{timestamp}.png')
acc_plot_path = os.path.join(plot_dir, f'acc_plot_{timestamp}.png')

# 绘制损失曲线
plt.figure()
plt.plot(range(1, 201), losses, label='Loss')  # 绘制损失曲线
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.savefig(loss_plot_path)  # 保存损失曲线图
plt.close()

# 绘制准确率曲线
plt.figure()
plt.plot(range(1, 201), train_accs, label='Train Acc')  # 绘制训练准确率曲线
plt.plot(range(1, 201), val_accs, label='Val Acc')  # 绘制验证准确率曲线
plt.plot(range(1, 201), test_accs, label='Test Acc')  # 绘制测试准确率曲线
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy')
plt.legend()
plt.savefig(acc_plot_path)  # 保存准确率曲线图
plt.close()

print(f"Loss plot saved to {loss_plot_path}")
print(f"Accuracy plot saved to {acc_plot_path}")