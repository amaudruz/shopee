import os
import torch
import random
from functools import partial
import PIL
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

class TripletDataset(Dataset) :
    def __init__(self, images_path, df, img_tfms, testing, text_tokenizer=None):
        super(TripletDataset, self).__init__()
        
        self.images_path = images_path
        self.img_tfms = img_tfms
        self.testing = testing
              
        self.df = df.copy()
        self.df['label_group'] = self.df['label_group'].astype('category').cat.codes
        self.df['index'] = range(self.df.shape[0])
        self.labels = self.df['label_group'].unique()
        self.label_to_index_list = self.df.groupby('label_group')['index'].apply(list)
        
    def __getitem__(self, index) :
        index_meta = self.df.iloc[index]
        
        anchor_image, anchor_text = self._get_item(index)
        
        if self.testing: return anchor_image, anchor_text
        
        label = index_meta['label_group']
        
        # positive sample
        pos_index = random.choice(self.label_to_index_list[label])
        # we don't want the positive sample being the same as the anchor
        while pos_index == index :
            pos_index = random.choice(self.label_to_index_list[label])
        pos_image, pos_text = self._get_item(pos_index)
        
        #negative sample
        neg_label = random.choice(self.labels)
        # Negative sample has to be different label from anchor 
        while neg_label == index :
            neg_label = random.choice(self.labels)
        neg_index = random.choice(self.label_to_index_list[neg_label])
        neg_image, neg_text = self._get_item(neg_index)
        
        return anchor_image, anchor_text, pos_image, pos_text, neg_image, neg_text
        
    def _get_item(self, index) :
        image = PIL.Image.open(os.path.join(self.images_path, 
                                            self.df.iloc[index]['image']))
        image = self.img_tfms(image)
        text = self.df.iloc[index]['title']
        return image, text
    
    def __len__(self) :
        return self.df.shape[0]

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
    dataset = TripletDataset(images_path, df_paths, img_tfms, testing)
    tokenizer = AutoTokenizer.from_pretrained(pretrianed_tokenizer)
    dl = DataLoader(dataset, batch_size=batch_size, collate_fn=partial(collate_fn, tokenizer), 
                    shuffle = shuffle, pin_memory = True)
    return dl

