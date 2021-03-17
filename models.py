from timm.models import layers
import torch
import torch.nn as nn
import timm
from torch.nn.modules.batchnorm import BatchNorm1d, BatchNorm2d
from transformers import DistilBertModel

class EmbedorNN(nn.Module) :
    def __init__(self, pretrained_image_embedor='resnet18', pretrained_text_embedor='distilbert-base-uncased',
                output_dim=128) :
        super(EmbedorNN, self).__init__()
        self.image_embedor = timm.create_model(pretrained_image_embedor, pretrained=True)
        self.image_pool = nn.AdaptiveAvgPool2d((1,1))
        self.text_embedor = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.head = nn.Linear(512+768, output_dim)
    
    def forward(self, x) :
        images, texts = x
        out_images = self.image_embedor.forward_features(images)
        out_images = self.image_pool(out_images).squeeze()
        out_text = self.text_embedor(texts['input_ids'], 
                                     attention_mask=texts['attention_mask'])[0][:,0,:]
        out = torch.cat([out_images, out_text], dim=-1)
        return self.head(out)

class Decidor(nn.Module) :
    def __init__(self, embedding_dim, dp=0.1):
        super().__init__()
        _layers = []
        _layers.append(nn.Linear(2*embedding_dim, 10))
        _layers.append(nn.BatchNorm1d(10))
        _layers.append(nn.ReLU())
        _layers.append(nn.Dropout(0.1))
        _layers.append(nn.Linear(10, 1))
        self.model = nn.Sequential(*layers)
    
    def forward(self, e1, e2) :
        return self.model(torch.cat([e1, e2], dim=-1))