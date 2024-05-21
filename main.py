import torch
from torch_geometric.datasets import WikipediaNetwork, Planetoid

# 加载Squirrel数据集
squirrel_dataset = WikipediaNetwork(root='/tmp/Squirrel', name='Squirrel')
squirrel_data = squirrel_dataset[0]

# 加载Cora数据集作为Coauthor CS Dataset的替代
cora_dataset = Planetoid(root='/tmp/Cora', name='Cora')
cora_data = cora_dataset[0]


from torch_geometric.nn import GATConv
import torch.nn.functional as F


class GAT(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, 8, heads=8, dropout=0.6)
        self.conv2 = GATConv(8 * 8, out_channels, heads=1, dropout=0.6)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(data, model, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data.x.to(device), data.edge_index.to(device))
    # 仅选择训练掩码对应的输出和标签进行损失计算
    train_mask = data.train_mask
    loss = F.nll_loss(out[train_mask], data.y[train_mask].to(device))
    loss.backward()
    optimizer.step()
    return loss.item()


def test(data, model):
    model.eval()
    logits, accs = model(data.x.to(device), data.edge_index.to(device)), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


# 实例化模型和优化器
model = GAT(cora_data.num_node_features, cora_dataset.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

# 训练和测试
for epoch in range(200):
    loss = train(cora_data, model, optimizer)
    train_acc, val_acc, test_acc = test(cora_data, model)
    print(f'Epoch {epoch+1}, Loss: {loss:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')

