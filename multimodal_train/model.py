from .imports import *

class EmbedorNN(nn.Module) :
    def __init__(self, pretrained_image_embedor, pretrained_text_embedor,
                 output_dim=512) :
        super(EmbedorNN, self).__init__()
        self.image_embedor = timm.create_model(pretrained_image_embedor, pretrained=True)
        self.image_pool = nn.AdaptiveAvgPool2d((1,1))
        self.text_embedor = BertModel.from_pretrained(pretrained_text_embedor)
        self.head = nn.Linear(2048+768, output_dim) #+768
        """self.head = nn.Sequential(
                                  #nn.ReLU(),
                                  #nn.Dropout(0.1),
                                  nn.Linear(2048+768, output_dim),
                                  #nn.Linear(output_dim // 2, output_dim)
                                 )"""
        
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
                                     attention_mask=texts['attention_mask']).pooler_output
        out = torch.cat([out_images, out_text], dim=-1)
        return self.head(out)

