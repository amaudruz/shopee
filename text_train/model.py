from .imports import *

class EMBBert(nn.Module) :
    def __init__(self, pretrained_text_embedor='bert-base-uncased',
                output_dim=512, agg='mean') :
        super(EMBBert, self).__init__()
        self.text_embedor = BertModel.from_pretrained(pretrained_text_embedor)
        self.agg = agg

    def forward(self, x) :
        texts = x
        
        out_text = self.text_embedor(texts['input_ids'], 
                                     attention_mask=texts['attention_mask'])
        if self.agg == 'mean' :
            out_text = out_text[0][:, 1:, :].mean(1)
        else : 
            out_text = out_text[0][:, 0, :]
        return out_text