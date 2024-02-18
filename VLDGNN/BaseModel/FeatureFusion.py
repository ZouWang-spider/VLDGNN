import torch
import torch.nn as nn

class FeatureFusion(nn.Module):
    def __init__(self):
        super(FeatureFusion, self).__init__()

    def forward(self, node1, node2):

        desired_size = 768
        pad_size = desired_size - node2.size(1)
        node2_reshaped = torch.cat([node2, torch.zeros(node2.size(0), pad_size)], dim=1)
        # 计算需要填充的大小
        pad_size = desired_size - node2_reshaped.size(0)
        # 填充原始张量的第一个维度
        if pad_size > 0:
            pad_tensor = torch.zeros(pad_size, node2_reshaped.size(1))
            padded_tensor = torch.cat([node2_reshaped, pad_tensor], dim=0)
        else:
            padded_tensor = node2_reshaped
        # print("node1_reshaped",padded_tensor.shape)   #(768,768)

        # 计算节点特征的点积
        interaction_matrix = torch.matmul(node1, padded_tensor.t())  # node1 * node2^T
        # print("interaction_matrix",interaction_matrix.shape)  #(10,768)
        return interaction_matrix


# # 随机生成一些张量作为输入
# input_size = 196
# hidden_size = 768
# node1 = torch.randn(10,768)  # 第二个节点的特征向量
# node2 = torch.randn(196,512)  # 第一个节点的特征向量
#
# # 实例化 FeatureFusion 类
# feature_fusion_model = FeatureFusion()
#
# # 执行前向传播
# hidden_vector = feature_fusion_model(node1, node2)
#
# # 打印输出张量的形状
# print("Output Shape:", hidden_vector.shape)