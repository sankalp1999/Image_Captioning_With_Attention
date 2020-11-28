import pandas as pd
import os
import spacy # for tokenizer
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from vocab import *

# I trained on kaggle so pathnames are different notebook and local

data_location =  'flickr8k'
caption_file = 'flickr8k/captions.txt'


'''
IMPORTANT POINTS RELATED TO CUSTOM DATASET

The dataloader workers work parallely. So, if you try to print, you will get very quick prints
that happen together.

__getitem__ packs, makes some processes that you want and sends it to the batch.
Then, further processing tweaks like packing sequences is to be done in Collate class
that is for dataloader.

e.g
    for idx, (img, caption, length) in enumerate(train_loader)

The arguments you receive here are determined by the __call__ (dunder method ?) function of the Collate class.

'''


class FlickrDataset(Dataset):
    def __init__(self, root_dir, captions_file, test, transform = None, freq_threshold = 2):
        self.root_dir = root_dir
        self.df = pd.read_csv(caption_file)
        self.transform = transform
        # Get images, caption column from pandas
        split_factor = 37500 # 4000/ 5 = reserving ~200 images for testing
        
        self.imgs = self.df["image"]
        self.imgs_test = self.imgs[split_factor:]
        self.imgs = self.imgs[0:split_factor]
        self.captions = self.df["caption"]
        self.captions_test = self.captions[split_factor:]
        self.captions = self.captions[0:split_factor]
        self.test = test
        #Init and Build vocab
        self.vocab = Vocab_Builder(freq_threshold) # freq threshold is experimental
        self.vocab.build_vocabulary(self.captions.tolist())

    def __len__(self):
        if (self.test == True):
            return len(self.imgs_test)
        
        return len(self.imgs)
    
    def __getitem__(self, index: int):

        # Indices are randomly sampled if Shuffle = True
        # otherwise sequentially.

        if self.test == False:
            caption = self.captions[index]
            img_id = self.imgs[index]
        elif self.test == True:
            index += 37500
            caption = self.captions_test[index]
            img_id = self.imgs_test[index]
        
        # Read the image corresponding to the index
        img = Image.open(os.path.join(self.root_dir, img_id)).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        
        numericalized_caption = [self.vocab.stoi["<SOS>"]] # stoi is string to index, start of sentence
        numericalized_caption += self.vocab.numericalize(caption) # Convert each word to a number in our vocab
        numericalized_caption.append(self.vocab.stoi["<EOS>"])

        #return tensor
        
        return img, torch.tensor(numericalized_caption)
    
    # It is recommended to make a separate validation dataloader. 
    
    @staticmethod
    def evaluation(self, index : int):
        caption = self.captions_test[index]
        img_id = self.imgs_test[index]

        # Read the image corresponding to the index
        img = Image.open(os.path.join(self.root_dir, img_id)).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)
        
        # Fixed BLEU score evaluation
        caption = self.vocab.tokenizer_eng(caption)
        
        return img, caption
# Caption lengths will be different, in our batch all have to be same length


'''
Goes to the dataloader
'''
class Collate:

    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        
        imgs = torch.cat(imgs, dim=0)
        
        targets = [item[1] for item in batch]
        
        lengths = torch.tensor([len(cap) for cap in targets]).long()
        
        targets = pad_sequence(targets, batch_first=True, padding_value=self.pad_idx)
        # I did not do batch_first = False in the beginning so I had to use torch.permute(1,0)
        # It ensures that shape : (batch_size, max_caption_length) 
        
        return imgs, targets, lengths

# caption file, Maybe change num_workers

def get_loader( root_folder,annotation_file,  transform, batch_size = 32,  num_workers = 8, shuffle = True, pin_memory = False, test = False):
    

    dataset =  FlickrDataset(root_folder,  annotation_file, test, transform = transform)
    pad_idx = dataset.vocab.stoi["<PAD>"]

    loader = DataLoader(
        dataset = dataset,
        batch_size = batch_size,
        num_workers = num_workers,
        shuffle = shuffle,
        pin_memory = pin_memory,
        collate_fn =  Collate(pad_idx = pad_idx)
    )

    return loader, dataset



