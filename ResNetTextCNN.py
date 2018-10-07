import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torchvision.models.resnet import BasicBlock,Bottleneck
import argparse
import math
import torch.utils.model_zoo as model_zoo

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class MyResNet(models.ResNet):
    def __init__(self,block,layers):
        super(MyResNet,self).__init__(block,layers)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x_flatten = x.view(x.size(0), -1) # 将（batch，outchanel,w,h）展平为（batch，outchanel*w*h）
        x = self.fc(x_flatten)

        return (x_flatten, x)
def ResnetX(pretrained=True,resnet_version=101,**kwargs):

    if resnet_version == 18:
        model = MyResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
        if pretrained:
            model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
        return model
    elif resnet_version == 34:
        model = MyResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
        if pretrained:
            model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
        return model
    elif resnet_version == 50:
        model = MyResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
        if pretrained:
            model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
        return model
    elif resnet_version == 101:
        model = MyResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
        if pretrained:
            model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
        return model
    elif resnet_version == 152:
        model = MyResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
        if pretrained:
            model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
        return model


class ResnetXTextCNN(nn.Module):
    def __init__(self,args):
        """
        :param num_classes:
        :param args:
        """
        super(ResnetXTextCNN, self).__init__()
        self.num_classes = args['cnn']['num_class']

        # ResnetX
        self.resnet_version = args['cnn']['resnet_version']

        # TextCNN
        self.vocb_size = args['text_cnn']['vocb_size']
        self.dim = args['text_cnn']['dim']
        self.n_class = args['text_cnn']['n_class']
        self.max_len = args['text_cnn']['max_len']
        self.embedding_matrix = torch.Tensor(args['text_cnn']['embedding_matrix'])
        self.device = args['device']

        self.build_ResnetX()
        self.build_textcnn()
        # self.set_fusion_layer()
    def build_ResnetX(self):
        self.resnetX = ResnetX(pretrained=True,resnet_version=101)

    def build_textcnn(self):
        #需要将事先训练好的词向量载入
        print(self.vocb_size,self.dim,self.embedding_matrix.shape)
        self.embeding = nn.Embedding(self.vocb_size, self.dim,_weight=self.embedding_matrix)
        self.conv1 = nn.Sequential(
                     nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
                     nn.ReLU(),
                     nn.MaxPool2d(kernel_size=2) # (16,64,64)
                     )
        self.conv2 = nn.Sequential(
                     nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
                     nn.ReLU(),
                     nn.MaxPool2d(2)
                     )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv4 = nn.Sequential(  # (16,64,64)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.out = nn.Linear(512, self.n_class)
    def output_from_fusion_layer(self,x_concate):
        in_features = x_concate.shape[1]
        self.fusion_layer = nn.Linear(in_features, self.num_classes)
        self.fusion_layer.to(self.device)
        return self.fusion_layer(x_concate)

    def forward(self, x_img, x_text):

        # ResnetX forward
        x_img,res = self.resnetX(x_img)

        # textCNN forward
        x_text = self.embeding(x_text)
        x_text = x_text.view(x_text.size(0),1,self.max_len,self.dim)
        x_text = self.conv1(x_text)
        x_text = self.conv2(x_text)
        x_text = self.conv3(x_text)
        x_text = self.conv4(x_text)
        x_text = x_text.view(x_text.size(0), -1) # 将（batch，outchanel,w,h）展平为（batch，outchanel*w*h）

        # concate x_img output and x_text output
        x_concate = torch.cat([x_img, x_text],1)
        output = self.output_from_fusion_layer(x_concate)
        return output

if __name__ == '__main__':
    model = ResnetXTextCNN()
    img_path = "../data/images_1000_560/1_1000_560.jpg"
    text_path = "../data/text_info/0/3/17099.txt"


