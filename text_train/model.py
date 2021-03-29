from .imports import *

class EMBBert(nn.Module) :
    def __init__(self, pretrained_text_embedor='bert-base-uncased',
                output_dim=512) :
        super(EMBBert, self).__init__()
        self.text_embedor = BertModel.from_pretrained(pretrained_text_embedor)
        self.head = nn.Sequential(nn.Linear(768, output_dim), 
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
            
    
    def forward(self, x) :
        texts = x
        
        out_text = self.text_embedor(texts['input_ids'], 
                                     attention_mask=texts['attention_mask'])[0][:,0,:]
        return F.normalize(self.head(out_text), dim=-1)