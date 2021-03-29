import os
import torch
import numpy as np
import random
from functools import partial
import PIL
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

class TripletDS(Dataset):
    def __init__(self, data, tokenizer, images_path, return_triplet = True):
        super().__init__()
        self.imgs = data['image'].tolist()
        self.unique_labels = data['label_group'].unique().tolist()
        self.labels = data['label_group'].tolist()
        self.label_to_index_dict = (data.reset_index(drop = True)
                                    .groupby('label_group')
                                    .apply(lambda x: x.index.tolist())
                                    .to_dict())
        self.texts = tokenizer(data['title'].values.tolist(), return_tensors = 'pt',
                               padding=True, truncation=True, max_length = 40)
        self.images_path = images_path
        self.return_triplet = return_triplet
        
    def __getitem__(self, idx):
        
        # anchor data
        anchor_label = self.labels[idx]
        anchor_img, anchor_txt = self._get_item(idx)
        
        if not self.return_triplet: return anchor_img, anchor_txt
        
        # neg data
        neg_label = np.random.choice(self.unique_labels)
        while neg_label == anchor_label:
            neg_label = np.random.choice(self.unique_labels)
        neg_idx = np.random.choice(self.label_to_index_dict[neg_label])
        neg_img, neg_txt = self._get_item(neg_idx)   
        
        # pos data
        pos_idxs = self.label_to_index_dict[anchor_label]
        # picking an index not equal to anchor's index
        pos_idxs = [o for o in pos_idxs if o != idx]
        
        if len(pos_idxs) == 0:
            # edge case, only 1 sample per label
            pos_idxs = [idx]
        pos_idx = np.random.choice(pos_idxs)
        pos_img, pos_txt = self._get_item(pos_idx)
        
        return anchor_img, anchor_txt, pos_img, pos_txt, neg_img, neg_txt
        
        
    def __len__(self):
        return len(self.imgs)
    
    def _get_item(self, idx):
        im = PIL.Image.open(os.path.join(self.images_path, self.imgs[idx]))
        im = torch.tensor(np.array(im) / 255.0, dtype = torch.float).permute(2,0,1)
        txt = {'input_ids' : self.texts['input_ids'][idx], 
               'attention_mask' : self.texts['attention_mask'][idx]}
        return im, txt

def text_to_device(text, device):
    return {'input_ids' : text['input_ids'].to(device),
            'attention_mask' : text['attention_mask'].to(device)}

def collate_fn(tokenizer, samples) :
    batch_size = len(samples)
    if len(samples[0]) == 2:
        images, texts = zip(*samples)
        images = torch.stack(images)
        texts = tokenizer(list(texts), padding=True, truncation=True, return_tensors="pt")
        return images, texts
    anchor_images, anchor_texts, pos_images, pos_texts, neg_images, neg_texts = zip(*samples)
    anchor_images = torch.stack(anchor_images)
    pos_images = torch.stack(pos_images)
    neg_images = torch.stack(neg_images)
    anchor_texts = tokenizer(list(anchor_texts), padding=True, truncation=True, return_tensors="pt")
    pos_texts = tokenizer(list(pos_texts), padding=True, truncation=True, return_tensors="pt")
    neg_texts = tokenizer(list(neg_texts), padding=True, truncation=True, return_tensors="pt")
    return anchor_images, anchor_texts, pos_images, pos_texts, neg_images, neg_texts

def create_dl(images_path, df_paths, img_tfms, pretrianed_tokenizer='distilbert-base-uncased', 
              batch_size=64, shuffle = True, testing = False) :
    dataset = TripletDS(images_path, df_paths, img_tfms, testing)
    tokenizer = AutoTokenizer.from_pretrained(pretrianed_tokenizer)
    dl = DataLoader(dataset, batch_size=batch_size, collate_fn=partial(collate_fn, tokenizer), 
                    shuffle = shuffle, pin_memory = True)
    return dl

