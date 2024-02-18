import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, CrossEntropyLoss

class FCLayer(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_aspect_tags, num_sentiment_classes,):
        super(FCLayer, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)  #396*10
        self.fc3 = nn.Linear(num_aspect_tags, num_sentiment_classes)  # 396,3

    def forward(self, x, aspect):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        vector = torch.matmul(x, aspect)   #396*2
        x = F.relu(self.fc3(vector))

        # 对输出进行池化，这里使用平均池化
        x_pooled = torch.mean(x, dim=0, keepdim=True)
        return F.softmax(x_pooled,dim=1)

# fc = nn.Linear(hidden_size2, num_sentiment_classes)
# sentiment = fc()



# hidden_size1 = 256
# hidden_size2 = 10
# num_aspect_tags = 2
# num_sentiment_classes = 3
# input_size = 768
# FC= FCLayer(input_size, hidden_size1, hidden_size2, num_aspect_tags, num_sentiment_classes)
#
# concatenated_output = torch.randn([396,768])
# finall_outpot = torch.randn([10, 2])
#
# sentiment_output = FC(concatenated_output,finall_outpot)
# print(sentiment_output.shape)

# # 使用 argmax 获取每行最大概率的类别索引
# print(sentiment_output)
# predicted_labels = torch.argmax(sentiment_output, dim=1)
# predicted_labels = torch.clamp(predicted_labels - 1, min=-1, max=1)
# # 找到张量中的众数
# mode_result = torch.mode(predicted_labels)
# predict_label = mode_result.values.item()
#
# # 打印结果
# print("预测的情感标签众数:", predict_label)


# # 真实情感标签
# sentiment = [1]
# labels = torch.tensor(sentiment, dtype=torch.long)  # 示例情感标签
#
# # 定义交叉熵损失
# criterion = nn.CrossEntropyLoss()
# # 计算交叉熵损失
# loss = criterion(sentiment_output, labels)
#
# # 打印结果
# print("交叉熵损失:", loss.item())


