import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class SharedSemanticSpace(nn.Module):
    def __init__(self, input_size1, input_size2, output_size3, shared_hidden_size, output_size):
        super(SharedSemanticSpace, self).__init__()
        self.mlp1 = MLP(input_size1, shared_hidden_size, output_size)
        self.mlp2 = MLP(input_size2, shared_hidden_size, output_size)
        self.mlp3 = MLP(output_size3, shared_hidden_size, output_size)
        self.fc = nn.Linear(output_size, 2)

    def forward(self, input1, input2):
        output1 = self.mlp1(input1)
        output2 = self.mlp2(input2)
        # 这里可以执行一些操作来组合两个输出，比如拼接或者求平均
        # 这里简单示范拼接
        combined_output = torch.matmul(output2, output1.t())
        # print(combined_output.shape)

        finall_outpot = self.mlp3(combined_output)
        # print(finall_outpot.shape)

        #FC,将维度torch.Size([10, 256])转化到torch.Size([10, 2])
        finall_outpot = self.fc(finall_outpot)
        output_vector = F.softmax(finall_outpot, dim=1)
        # print(output_vector.shape)

        return output_vector



#
# # 示例数据
# input_size1 = 512
# input_size2 = 768
# shared_hidden_size = 256
# output_size = 256
# output_size3 = 196
#
# # 创建模型
# mlp1 = MLP(input_size1, shared_hidden_size, output_size)
# mlp2 = MLP(input_size2, shared_hidden_size, output_size)
# # mlp3 = MLP(196, shared_hidden_size, output_size)
# shared_semantic_space = SharedSemanticSpace(output_size, output_size, output_size3, shared_hidden_size, output_size)
#
# # 随机生成一些示例输入
# input_tensor1 = torch.randn(196, input_size1)
# input_tensor2 = torch.randn(10, input_size2)
#
# # 通过各自的MLP进行学习
# output_tensor1 = mlp1(input_tensor1)
# output_tensor2 = mlp2(input_tensor2)
#
# # 将学到的隐层向量输入到共享语义子空间
# finall_outpot = shared_semantic_space(output_tensor1, output_tensor2)
#
# # 输出形状
# print("Combined Output Shape:", finall_outpot.shape)
