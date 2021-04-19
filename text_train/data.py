from .imports import *

class TextDS(Dataset):
    def __init__(self, data, tokenizer, max_length=200):
        super().__init__()
        self.unique_labels = data['label_group'].unique().tolist()
        self.labels = data['label_group'].astype('category')
        self.label_codes = self.labels.cat.codes
        
        self.texts = tokenizer(data['title'].values.tolist(), return_tensors = 'pt',
                               padding=True, truncation=True, max_length = max_length)
        
        
    def __getitem__(self, idx):
        
        # anchor data
        label = self.label_codes.iloc[idx] 
        txt = self._get_item(idx)
        
        return txt, label
        
        
    def __len__(self):
        return len(self.labels)
    
    def _get_item(self, idx):
        txt = {'input_ids' : self.texts['input_ids'][idx], 
               'attention_mask' : self.texts['attention_mask'][idx],
               'token_type_ids' : self.texts['token_type_ids'][idx]}
        return txt

def load_data(df_path='data/train.csv', train_perc=0.7) :
    df = pd.read_csv(df_path)
    n_classes = df['label_group'].nunique()

    # train val split
    n_train_examples = int(train_perc * len(df))

    train_df = df.iloc[:n_train_examples]
    val_df = df.iloc[n_train_examples:]

    return df, train_df, val_df

def text_to_device(text, device):
    return {'input_ids' : text['input_ids'].to(device),
            'attention_mask' : text['attention_mask'].to(device),
            'token_type_ids' : text['token_type_ids'].to(device)}