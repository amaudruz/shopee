from .imports import *

class MultiDS(Dataset):
    def __init__(self, data, tokenizer, images_path, return_triplet = True):
        super().__init__()
        self.imgs = data['image'].tolist()
        self.labels = data['label_group'].tolist()
        self.texts = tokenizer(data['title'].values.tolist(), return_tensors = 'pt',
                               padding=True, truncation=True, max_length = 30)
        self.images_path = images_path
        
    def __getitem__(self, idx):
        
        # anchor data
        anchor_label = self.labels[idx]
        anchor_img, anchor_txt = self._get_item(idx)
        return anchor_img, anchor_txt, torch.tensor(anchor_label)
        
    def __len__(self):
        return len(self.imgs)
    
    def _get_item(self, idx):
        im = PIL.Image.open(os.path.join(self.images_path, self.imgs[idx]))
        #im = im.resize((500,500))
        im = torch.tensor(np.array(im) / 255.0, dtype = torch.float).permute(2,0,1)
        txt = {'input_ids' : self.texts['input_ids'][idx], 
               'attention_mask' : self.texts['attention_mask'][idx]}
        return im, txt

def load_data(df_path='data/train.csv', train_perc=0.7) :
    df = pd.read_csv(df_path)
    n_classes = df['label_group'].nunique()

    # train val split
    n_train_examples = int(train_perc * len(df))

    train_df = df.iloc[:n_train_examples]
    val_df = df.iloc[n_train_examples:]

    return df, train_df, val_df

def create_dl(df, images_dir, batch_size=64, shuffle=True) :
    tr_ds = MultiDS(df, images_dir)
    tr_dl = DataLoader(tr_ds, batch_size = batch_size, shuffle = shuffle, pin_memory = True)
    return tr_dl
