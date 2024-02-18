import nltk
import numpy as np
import torch
import torch.nn as nn
from torchcrf import CRF
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.nn import Linear, CrossEntropyLoss
from sklearn.metrics import log_loss

from VisionLanguageMABSA.ImagetoGraph.ImagePatch import image_patches
from VisionLanguageMABSA.ImagetoGraph.ImageFeature import patches_features, Neighbors, create_Graph
from VisionLanguageMABSA.TexttoGraph.BiAffine import BiAffine, BERT_Embedding
from VisionLanguageMABSA.BaseModel.GCN import GCNModel
from VisionLanguageMABSA.BaseModel.GCN import MultiheadAttentionLayer
from VisionLanguageMABSA.BaseModel.NeuralTensorNetwork import NeuralTensorNetwork
from VisionLanguageMABSA.BaseModel.FeatureFusion import FeatureFusion
from VisionLanguageMABSA.DataProcess.DataProcess import load_dataset
from VisionLanguageMABSA.BaseModel.FCLayer import FCLayer
from VisionLanguageMABSA.BaseModel.MLP import MLP, SharedSemanticSpace
from VisionLanguageMABSA.BaseModel.FeatureInteraction import FeatureInteraction




data_path = "E:/PythonProject2/VisionLanguageMABSA/Datasets/twitter2015/train.txt"
image_path = "E:/PythonProject2/VisionLanguageMABSA/Datasets/twitter2015_images"
samples = load_dataset(data_path,image_path)


def convert_aspect_terms_to_binary_sequence(sentence, aspect_terms):
    # 使用nltk库进行分词
    tokens = nltk.word_tokenize(sentence)

    # 创建一个与tokens相同长度的二进制序列，初始值为0
    binary_sequence = [0] * len(tokens)

    # 将方面词的位置标记为1

    start_idx = sentence.find(aspect_terms)
    end_idx = start_idx + len(aspect_terms)
    aspect_tokens = nltk.word_tokenize(sentence[:start_idx]) + nltk.word_tokenize(aspect_terms)
    start_token_idx = len(nltk.word_tokenize(' '.join(aspect_tokens[:-1])))
    end_token_idx = start_token_idx + len(nltk.word_tokenize(aspect_terms))

    # 将方面词对应的位置标记为1
    binary_sequence[start_token_idx-1:end_token_idx-1] = [1] * (end_token_idx - start_token_idx)


    return binary_sequence

#CRF layer
class CRFModule(nn.Module):
    def __init__(self, num_tags):
        super(CRFModule, self).__init__()
        self.crf = CRF(num_tags)

    def forward(self, inputs):
        return self.crf(inputs)

def get_dataset(sample):
    image_path = sample['image_path']
    sentence = sample['sentence']
    aspect_terms = sample['aspect_term']
    sentiments = sample['sentiment']
    return image_path, sentence, aspect_terms, sentiments


def image_feature_func(image_path):
    # 使用函数获取 patches
    patches = image_patches(image_path, (14, 14))

    # 提取特征向量
    image_feature = patches_features(patches)
    # print(image_feature.shape[1])
    # 查看特征向量的形状
    # print("Image:", image_feature.shape)  # (196,512)
    # print(image_feature)
    # KNN算法寻找当前特征向量的邻居，8个
    k_neighbors = 8
    indices = Neighbors(k_neighbors, image_feature)
    # 根据邻居节点构建图
    image_feature = torch.tensor(image_feature)
    image_graph = create_Graph(indices)
    return image_feature, image_graph, patches


def text_feature_func(sentence):
    # part of Language
    text = nltk.word_tokenize(sentence)
    text_graph, rels, pos = BiAffine(sentence)
    word_feature, dependency_feature, pos_feature = BERT_Embedding(sentence, rels, pos)
    syn_feature = torch.cat((word_feature, dependency_feature, pos_feature), dim=1)
    #Pooling k=3
    max_pooling = nn.MaxPool1d(kernel_size=3)
    text_feature = max_pooling(syn_feature)


    # print("Text:", word_embedding_feature.shape)  # (6,768)
    # print(word_embedding_feature)
    return text_feature, text_graph, text


def image_GCN(image_feature, patches):
    # Image GCN parameter
    input_size1 = image_feature.shape[1]
    hidden_size1 = image_feature.shape[1]  # 隐藏层的大小
    num_layers1 = 2  # GCN layer
    num_node1 = len(patches)

    # image GCN模型和 text GCN模型
    image_gcn = GCNModel(input_size1, hidden_size1, num_layers1, num_node1)
    return image_gcn


def text_GCN(word_embedding_feature, text):
    # Text GCN parameter
    input_size2 = word_embedding_feature.shape[1]
    hidden_size2 = word_embedding_feature.shape[1]  # 隐藏层的大小
    num_layers2 = 2  # GCN layer
    num_node2 = len(text)

    text_gcn = GCNModel(input_size2, hidden_size2, num_layers2, num_node2)
    return text_gcn


def attention(image_feature, word_embedding_feature):
    # Attention
    hidden_size1 = image_feature.shape[1]
    hidden_size2 = word_embedding_feature.shape[1]
    num_heads = 8
    image_attention = MultiheadAttentionLayer(hidden_size1, num_heads)
    text_attention = MultiheadAttentionLayer(hidden_size2, num_heads)
    return image_attention, text_attention


def NeuralTensor(image_feature, word_embedding_feature):
    # Neural tensor Network
    text_Tensor = word_embedding_feature.shape[0]
    text_Tensor2 = word_embedding_feature.shape[1]
    image_Tensor = image_feature.shape[0]
    image_Tensor2 = image_feature.shape[1]
    NeuralTensormodel = NeuralTensorNetwork(text_Tensor, text_Tensor2, image_Tensor, image_Tensor2)
    return NeuralTensormodel


def FeaturesFusion(word_embedding_feature):
    # Node Feature Interaction
    input_NodeFeature = word_embedding_feature.shape[0]  # 输入向量的大小
    hidden_NodeFeature = word_embedding_feature.shape[1]  # 隐层向量的大小
    FeatureInteraction = FeatureFusion()
    return FeatureInteraction



hidden_size1 = 256
num_aspect_tags = 2
num_sentiment_classes = 3
def FC_func(word_embedding_feature):
    hidden_size2 = word_embedding_feature.shape[0]
    input_size = word_embedding_feature.shape[1]
    fcmodel= FCLayer(input_size, hidden_size1, hidden_size2, num_aspect_tags, num_sentiment_classes)
    return  fcmodel


#model parameter
num_heads = 8
# 定义各个组件的学习率
learning_rate = 0.00002

def modeltrain(image_feature, image_graph, word_embedding_feature, text_graph, aspect_term_seq, sentiments,
               image_gcn, text_gcn, image_attention, text_attention, neuraltensormodel, featuresfusion, featureinteraction, FCmodel):
    # 训练使用
    # Image Train
    imagegcn_output = image_gcn(image_feature, image_graph)
    # print(imagegcn_output.shape)   #torch.Size([196, 512])
    gcn_output1 = imagegcn_output.view(imagegcn_output.shape[0], 1, imagegcn_output.shape[1])  # 196个patches，512为对应维度
    image_att = image_attention(gcn_output1)
    imageatt_output = image_att.squeeze(dim=1)
    # print(imageatt_output.shape)   torch.Size([196, 512])

    # Text Train
    textgcn_output = text_gcn(word_embedding_feature, text_graph)
    # print(textgcn_output.shape)   torch.Size([10, 768])
    gcn_output2 = textgcn_output.view(textgcn_output.shape[0], 1, textgcn_output.shape[1])  # 196个patches，512为对应维度
    text_att = text_attention(gcn_output2)
    textatt_output = text_att.squeeze(dim=1)
    # print(textatt_output.shape)   torch.Size([10, 768])

    # 处理张量维度一致 ([196, 512])==>([196, 768])
    desired_size = 768
    pad_size = desired_size - imagegcn_output.size(1)
    imagegcn_reshaped = torch.cat([imagegcn_output, torch.zeros(imagegcn_output.size(0), pad_size)], dim=1)
    # Module learning
    output1 = neuraltensormodel(textgcn_output, imagegcn_reshaped)  #word_embedding_feature, image_feature

    output2 = featuresfusion(textatt_output, imageatt_output)


    #Feature Interaction
    # 转换image的维度(196,512) ===>(196,768)
    desired_size = 768
    pad_size = desired_size - imageatt_output.size(1)
    imageatt_reshaped = torch.cat([imageatt_output, torch.zeros(imageatt_output.size(0), pad_size)], dim=1)
    # print(image_reshaped.shape)  # (196,768)

    final_output = featureinteraction(textatt_output, imageatt_reshaped, imageatt_reshaped)
    # print(final_output.shape)

    NNmodel = nn.Sequential(
        nn.Linear(768, 2),
        nn.Softmax(dim=1)
    )

    output_tensor = NNmodel(final_output) #(10,2)

    # CRF 层 输出方面词的个数
    concatenated_batch = output_tensor.unsqueeze(0)  #二维转化为三维
    # print(concatenated_batch.shape)  # torch.Size([1, 10, 2])
    num_aspect_tags = 2
    crf_layer = CRF(num_tags=num_aspect_tags, batch_first=True)
    predicted_aspect_tags = crf_layer.decode(concatenated_batch)   #解码返回序列
    predict_label = torch.tensor(predicted_aspect_tags).view(-1).tolist()

    # 检查长度是否相等
    if len(predict_label) == len(aspect_term_seq):
        pass
    elif len(predict_label) < len(aspect_term_seq):
        aspect_term_seq = aspect_term_seq[-len(predict_label):]
    else:  # len(predict_label) > len(aspect_term_seq)
        aspect_term_seq = [0] * (len(predict_label) - len(aspect_term_seq)) + aspect_term_seq

    # print(predict_label)  # [0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    # print(aspect_term_seq)  # [0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1]

    # CRF通常使用负对数似然作为损失,对数似然损失（Log-Likelihood Loss）
    aspect_loss = log_loss(aspect_term_seq, predict_label)
    # print(aspect_loss)

    # 全连接层  输出方面词对应的情感极性
    concatenated_output = torch.cat([output1, output2], dim=0)   #(20,768)
    sentiment_output = FCmodel(concatenated_output, output_tensor)
    # print(sentiment_output)

    # 映射标签到类别索引
    sentiment = [sentiments]
    labels = torch.tensor(sentiment, dtype=torch.long)  # 情感标签
    label_mapping = {-1: 0, 0: 1, 1: 2}
    mapped_labels = torch.tensor([label_mapping[label.item()] for label in labels])
    # print(mapped_labels)
    # 计算损失（例如，使用交叉熵损失）
    criterion = CrossEntropyLoss()
    aspect_sentiment_loss = criterion(sentiment_output, mapped_labels)
    # print(aspect_sentiment_loss)

    # 计算总的损失
    total_loss = aspect_loss + aspect_sentiment_loss

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss.item()}, aspect_loss:{aspect_loss}, aspect_sentiment_loss:{aspect_sentiment_loss}')

    # Set up the optimizer
    # 合并所有参数和优化器
    all_parameters = list(image_gcn.parameters()) + list(text_gcn.parameters()) + list(image_attention.parameters()) + \
                     list(image_attention.parameters()) + list(text_attention.parameters()) + list(
        neuraltensormodel.parameters()) + list(featuresfusion.parameters())

    optimizer = Adam(all_parameters, lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    # Backward pass and optimization step
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    # Learning rate scheduling
    scheduler.step()




if __name__ == "__main__":
    num_epochs = 50
    batch_size = 32

    for epoch in range(num_epochs):
        # Iterate through batches
        for start_idx in range(0, len(samples), batch_size):
            end_idx = start_idx + batch_size
            batch_samples = samples[start_idx:end_idx]

            # datasets Train
            for sample in batch_samples:
                image_path, sentence, aspect_terms, sentiments = get_dataset(sample)  # get dataset,image and text
                # 转化为序列形式
                aspect_term_seq = convert_aspect_terms_to_binary_sequence(sentence, aspect_terms)
                image_feature, image_graph, patches = image_feature_func(image_path)  # image feature
                word_embedding_feature, text_graph, text = text_feature_func(sentence)  # text feature
                image_gcn = image_GCN(image_feature, patches)  # construct model
                text_gcn = text_GCN(word_embedding_feature, text)  # construct model
                image_attention, text_attention = attention(image_feature, word_embedding_feature)  # attention
                neuraltensormodel = NeuralTensor(image_feature, word_embedding_feature)
                featuresfusion = FeaturesFusion(word_embedding_feature)
                # mlp1, mlp2, shared_semantic_space = SharedSemanticlayer(image_feature, word_embedding_feature)
                featureinteraction = FeatureInteraction(768)
                FCmodel = FC_func(word_embedding_feature)
                modeltrain(image_feature, image_graph, word_embedding_feature, text_graph, aspect_term_seq, sentiments,
                           image_gcn, text_gcn, image_attention, text_attention, neuraltensormodel, featuresfusion, featureinteraction, FCmodel)

















