from .imports import *

class ImageDS(Dataset):
    def __init__(self, data, images_path, return_triplet = True, trans = None):
        super().__init__()
        self.imgs = data['image'].tolist()
        #tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
        self.unique_labels = data['label_group'].unique().tolist()
        """self.texts = tokenizer(data['title'].values.tolist(), return_tensors = 'pt',
                               padding=True, truncation=True, max_length = 200)"""
        self.labels = data['label_group'].astype('category')
        self.label_codes = self.labels.cat.codes
        self.trans = trans
        
        self.images_path = images_path
        
    def __getitem__(self, idx):
        
        img = self._get_item(idx)
        label = self.label_codes.iloc[idx]
        return img, label
    def __len__(self):
        return len(self.imgs)
    
    def _get_item(self, idx):
        im = PIL.Image.open(os.path.join(self.images_path, self.imgs[idx])).convert('RGB')
        if self.trans is not None:
            im = self.trans(im)
        #im = torch.tensor(np.array(im) / 255.0, dtype = torch.float).permute(2,0,1)
        """txt = {'input_ids' : self.texts['input_ids'][idx], 
               'attention_mask' : self.texts['attention_mask'][idx]}"""
        return im#np.array(im) / 255.0#, txt

def load_data(df_path='data/train.csv', train_perc=0.7) :
    df = pd.read_csv(df_path)
    n_classes = df['label_group'].nunique()

    # train val split
    n_train_examples = int(train_perc * len(df))

    train_df = df.iloc[:n_train_examples]
    val_df = df.iloc[n_train_examples:]

    return df, train_df, val_df

def create_dl(df, images_dir, batch_size=64, shuffle=True, trans = None) :
    tr_ds = ImageDS(df, images_dir, trans = trans)
    tr_dl = DataLoader(tr_ds, batch_size = batch_size, shuffle = shuffle, pin_memory = True)
    return tr_dl
