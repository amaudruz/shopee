from timm.models import layers
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torch.nn.modules.batchnorm import BatchNorm1d, BatchNorm2d
from transformers import BertModel
import numpy as np

class EmbedorNN(nn.Module) :
    def __init__(self, pretrained_image_embedor='resnet50', pretrained_text_embedor='bert-base-uncased',
                output_dim=512) :
        super(EmbedorNN, self).__init__()
        self.image_embedor = timm.create_model(pretrained_image_embedor, pretrained=True)
        self.image_pool = nn.AdaptiveAvgPool2d((1,1))
        self.text_embedor = BertModel.from_pretrained(pretrained_text_embedor)
        self.head = nn.Sequential(nn.Linear(2048+768, output_dim), 
                                  #nn.ReLU(), 
                                  #nn.Linear(1024, output_dim)
                                 )
        
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                sz = m.weight.data.size(-1)
                m.weight.data.normal_(mean=0.0, std=1/np.sqrt(sz))
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
                m.bias.data.zero_()
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()
            
    def freeze_lm(self):
        for parameter in self.text_embedor.parameters():
            parameter.requires_grad = False
            
    def unfreeze_lm(self):
        for parameter in self.text_embedor.parameters():
            parameter.requires_grad = True
            
    def freeze_cnn(self):
        for parameter in self.image_embedor.parameters():
            parameter.requires_grad = False
            
    def unfreeze_cnn(self):
        for parameter in self.image_embedor.parameters():
            parameter.requires_grad = True
    
    def forward(self, x) :
        images, texts = x
        out_images = self.image_embedor.forward_features(images)
        out_images = self.image_pool(out_images).squeeze()
        out_text = self.text_embedor(texts['input_ids'], 
                                     attention_mask=texts['attention_mask'])[0][:,0,:]
        out = torch.cat([out_images, out_text], dim=-1)
        return F.normalize(self.head(out), dim=-1)

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