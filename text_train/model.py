from .imports import *

class EMBBert(nn.Module) :
    def __init__(self, pretrained_text_embedor='bert-base-uncased',
                output_dim=512) :
        super(EMBBert, self).__init__()
        self.text_embedor = BertModel.from_pretrained(pretrained_text_embedor)

    def forward(self, x) :
        texts = x
        
        out_text = self.text_embedor(texts['input_ids'], 
                                     attention_mask=texts['attention_mask'])[0].mean(1)
        return out_text