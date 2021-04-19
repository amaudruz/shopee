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
                                     attention_mask=texts['attention_mask'],
                                     token_type_ids = texts['token_type_ids'])
        if self.agg == 'mean' :
            out_text = (out_text[0]*texts['attention_mask'].unsqueeze(-1))[:, 1:, :].sum(1) / (texts['attention_mask'].sum(1)-1).unsqueeze(-1)
        elif self.agg == 'mean_dumb': 
            out_text = out_text[0].mean(1)
        elif self.agg == 'cls' :
            out_text = out_text[0][:, 0, :]
        else :
            raise NotImplementedError()
        return out_text