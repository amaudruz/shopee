from .imports import *

class EMBRes(nn.Module) :
    def __init__(self, pretrained_image_embedor='resnet50',
                output_dim=512) :
        super(EMBRes, self).__init__()
        self.image_embedor = timm.create_model(pretrained_image_embedor, pretrained=True)
        self.image_pool = nn.AdaptiveAvgPool2d((1,1))
        self.head = nn.Sequential(nn.Linear(2048, output_dim), 
                                  #nn.ReLU(),
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
    
    def _get_embs(self, x) :
        images = x
        out_images = self.image_embedor.forward_features(images)
        out_images = self.image_pool(out_images).squeeze()
        #return F.normalize(out_images, dim=-1)
        return out_images
    
    def forward(self, x) :
        out_images = self._get_embs(x)
        
        return self.head(out_images)