from imports import *

class MultiDS(Dataset):
    def __init__(self, data, images_path, 
                 context_length=77, column='translated_titles'):
        super().__init__()
        self.imgs = data['image'].tolist()
        self.unique_labels = data['label_group'].unique().tolist()
        self.texts = clip.tokenize(data[column].values.tolist(), context_length=context_length, truncate=True)
        self.labels = data['label_group'].astype('category')
        self.label_codes = self.labels.cat.codes
        
        self.images_path = images_path
        
    def __getitem__(self, idx):
        
        img, txt = self._get_item(idx)
        label = self.label_codes.iloc[idx]
        return img, txt, label
    def __len__(self):
        return len(self.imgs)
    
    def _get_item(self, idx):
        im = PIL.Image.open(os.path.join(self.images_path, self.imgs[idx])).convert('RGB')
        im = torch.tensor(np.array(im) / 255.0, dtype = torch.float).permute(2,0,1)
        txt = self.texts[idx]
        return im, txt

class TextDS(Dataset):
    def __init__(self, data, context_length=77, column='translated_titles'):
        super().__init__()
        self.unique_labels = data['label_group'].unique().tolist()
        self.labels = data['label_group'].astype('category')
        self.label_codes = self.labels.cat.codes
        
        self.texts = clip.tokenize(data[column].values.tolist(), context_length=context_length, truncate=True)
        
        
    def __getitem__(self, idx):
        
        # anchor data
        label = self.label_codes.iloc[idx] 
        txt = self._get_item(idx)
        
        return txt, label
        
        
    def __len__(self):
        return len(self.labels)
    
    def _get_item(self, idx):
        txt = self.texts[idx]
        return txt