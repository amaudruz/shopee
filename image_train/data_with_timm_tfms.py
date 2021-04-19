from .imports import *

class ImageDS(Dataset):
    def __init__(self, data, images_path, trans = None):
        super().__init__()
        self.imgs = data['image'].tolist()
        self.unique_labels = data['label_group'].unique().tolist()
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
        return im

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
