import torch
import torch.nn as nn


class NeuralTensorNetwork(nn.Module):
    def __init__(self, text_Tensor, text_Tensor2, image_Tensor, image_Tensor2,):
        super(NeuralTensorNetwork, self).__init__()
        self.text_tensor = text_Tensor2
        # 奇异值分解的参数
        self.W = nn.Parameter(torch.randn(text_Tensor2, image_Tensor))  #[768,196]
        # 拼接的参数
        self.V = nn.Parameter(torch.randn(text_Tensor, text_Tensor+image_Tensor)) #[10,(196+10)]
        # 偏置项
        self.b = nn.Parameter(torch.zeros(text_Tensor,text_Tensor2)) #[10,768] image_Tensor,text_Tensor2

    def forward(self, text_h, image_reshaped):
        # 奇异值分解部分
        interaction_part = torch.matmul(torch.matmul(text_h, self.W), image_reshaped).squeeze()
        print(interaction_part.shape)   #([10,768])
        # # 处理张量维度一致
        # desired_size = 768
        # pad_size = desired_size - image_h.size(1)
        # print(pad_size)
        # tensor1_reshaped = torch.cat([image_h, torch.zeros(image_h.size(0), pad_size)], dim=1)

        # 在垂直方向上拼接
        concatenate_vector = torch.cat([text_h,image_reshaped], dim=0)
        concat_part = torch.matmul(self.V, concatenate_vector)
        # 总体输出
        output = torch.sigmoid(interaction_part + concat_part + self.b)
        return output



# # 随机生成一些张量作为输入
# text_Tensor = 10
# text_Tensor2 = 768
# image_Tensor = 196
# image_Tensor2 = 512
#
# image_h = torch.randn(image_Tensor,image_Tensor2)  # 图像张量
# text_h = torch.randn(text_Tensor,text_Tensor2)  # 文本张量
#
# # 处理张量维度一致
# desired_size = 768
# pad_size = desired_size - image_h.size(1)
# image_reshaped = torch.cat([image_h, torch.zeros(image_h.size(0), pad_size)], dim=1)
# print(image_reshaped.shape)
#
# # 实例化神经张量网络
# neural_tensor_network = NeuralTensorNetwork(text_Tensor, text_Tensor2, image_Tensor, image_Tensor2)
#
# # 执行前向传播
# output = neural_tensor_network(text_h, image_reshaped)
#
# # 打印输出张量的形状
# print("Output Shape:", output.shape)  #(10,768)