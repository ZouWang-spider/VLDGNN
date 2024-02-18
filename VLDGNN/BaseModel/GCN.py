import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as geo_nn
from torch.nn import MultiheadAttention

class GCNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_node):
        super(GCNModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gcn_layers = nn.ModuleList()
        self.gcn_layers.append(geo_nn.GCNConv(input_size, hidden_size))  # 第一层使用 GCNConv

        for _ in range(1, num_layers):
            self.gcn_layers.append(geo_nn.GCNConv(hidden_size, hidden_size))  # 后续层继续使用 GCNConv

        self.fc = nn.Linear(hidden_size, num_node)  # 分类头

    def forward(self, node_features, edge_index):

        for layer in self.gcn_layers:
            node_features = layer(node_features, edge_index)

        # # 使用平均池化操作将单词级别的输出汇总为句子级别的输出
        # sentence_output = torch.mean(node_features, dim=0, keepdim=True)

        return node_features


class MultiheadAttentionLayer(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(MultiheadAttentionLayer, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads)

    def forward(self, input_tensor):
        # 注意：MultiheadAttention 输入要求为 (seq_len, batch_size, embed_dim)
        input_tensor = input_tensor.permute(1, 0, 2)
        attention_output, _ = self.attention(input_tensor, input_tensor, input_tensor)
        # 注意：输出维度为 (seq_len, batch_size, embed_dim)
        attention_output = attention_output.permute(1, 0, 2)
        return F.relu(attention_output)
