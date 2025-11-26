# gnn_vuln_scanner/multilabel_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class MultiLabelVulnGNN(nn.Module):
    """多标签漏洞检测GNN"""

    def __init__(self, input_dim=13, hidden_dim=128, num_classes=10):
        super(MultiLabelVulnGNN, self).__init__()

        # 图卷积层
        self.convs = nn.ModuleList([
            GCNConv(input_dim, hidden_dim),
            GCNConv(hidden_dim, hidden_dim),
            GCNConv(hidden_dim, hidden_dim)
        ])

        # 批归一化
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        ])

        # 分类器（为每个漏洞类型输出一个logit）
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # 图卷积 + 批归一化
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=0.3, training=self.training)

        # 全局池化
        x = global_mean_pool(x, batch)

        # 多标签分类
        logits = self.classifier(x)

        return logits
