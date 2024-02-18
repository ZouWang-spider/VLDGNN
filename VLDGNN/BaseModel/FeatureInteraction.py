import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureInteraction(nn.Module):
    def __init__(self, hidden_dim):
        super(FeatureInteraction, self).__init__()
        self.query_layer = nn.Linear(hidden_dim, hidden_dim)
        self.key_layer = nn.Linear(hidden_dim, hidden_dim)
        self.value_layer = nn.Linear(hidden_dim, hidden_dim)

        # 假设这里定义了注意力层和MLP层的参数

        self.mlp_layer = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim)
        )

        # Layer Normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self,query, key, value):
        # 将输入张量通过权重矩阵变换
        query_transformed = self.query_layer(query)
        key_transformed = self.key_layer(key)
        value_transformed = self.value_layer(value)

        # 注意力机制
        attn_weights = torch.matmul(query_transformed, key_transformed.transpose(0, 1))
        attn_weights = F.softmax(attn_weights, dim=-1)

        # 使用注意力权重对 Value 进行加权求和
        output = torch.matmul(attn_weights, value_transformed)
        # print(output.shape) #（10,768）

        # Add & Layer Normalization
        attention_output = self.layer_norm(output + query)

        # MLP
        mlp_output = self.mlp_layer(attention_output)

        # 另一个 Add & Layer Normalization
        final_output = self.layer_norm(mlp_output + attention_output)
        # print(final_output.shape) #（10,768）

        return final_output


# # 使用示例
# text = torch.randn(10, 768)  #Query
# image = torch.randn(196, 512)  #Key,Value
#
# #转换image的维度
# desired_size = 768
# pad_size = desired_size - image.size(1)
# image_reshaped = torch.cat([image, torch.zeros(image.size(0), pad_size)], dim=1)
# print(image_reshaped.shape)   #(196,768)
#
# featureinteraction = FeatureInteraction(768)
# final_output = featureinteraction(text, image_reshaped, image_reshaped)

