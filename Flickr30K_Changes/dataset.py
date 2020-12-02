import os
import pandas as pd
import spacy # for tokenizer
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

# python -m spacy download en
spacy_eng = spacy.load('en_core_web_sm')

class Vocab_Builder:
    
    def __init__ (self,freq_threshold):

        # freq_threshold is to allow only words with a frequency higher 
        # than the threshold

        self.itos = {0 : "<PAD>", 1 : "<SOS>", 2 : "<EOS>", 3 : "<UNK>"}  #index to string mapping
        self.stoi = {"<PAD>" : 0, "<SOS>" : 1, "<EOS>" : 2, "<UNK>" : 3}  # string to index mapping
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)
    
    @staticmethod
    def tokenizer_eng(text):
        #Removing spaces, lower, general vocab related work

        return [token.text.lower() for token in spacy_eng.tokenizer(text)]
    
    def build_vocabulary(self, sentence_list):
        frequencies = {} # dict to lookup for words
        idx = 4

        # FIXME better ways to do this are there
        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies:
                    frequencies[word] = 1
                else:
                    frequencies[word] += 1 
                if(frequencies[word] == self.freq_threshold):
                    #Include it
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1
    
    # Convert text to numericalized values
    def numericalize(self,text):
        tokenized_text = self.tokenizer_eng(text) # Get the tokenized text
        
        # Stoi contains words which passed the freq threshold. Otherwise, get the <UNK> token
        return [self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
        for token in tokenized_text ]
    
    def denumericalize(self, tensors):
        text = [self.itos[token] if token in self.itos else self.itos[3]]
        return text

class FlickrDataset(Dataset):
    def __init__(self, root_dir, captions_file, test, transform = None, freq_threshold = 5):
        self.root_dir = root_dir
        self.df = pd.read_csv(caption_file , delimiter='|')
        self.transform = transform
        # Get images, caption column from pandas
        self.split_factor = 153915 # 4000/ 5 = reserving 200 images for testing
        
        self.imgs = self.df["image_name"]
        self.imgs_test = self.imgs[self.split_factor:]
        self.imgs = self.imgs[0:self.split_factor]
        self.captions = self.df["caption_text"]
        self.captions_test = self.captions[self.split_factor:]
        self.captions = self.captions[0:self.split_factor]
        self.test = test
        #Init and Build vocab
        self.vocab = Vocab_Builder(freq_threshold) # freq threshold is experimental
        self.vocab.build_vocabulary(self.captions.tolist())

    def __len__(self):
        if (self.test == True):
            return len(self.imgs_test)
        
        return len(self.imgs)
    
    def __getitem__(self, index: int):
        
        if self.test == False:
            caption = self.captions[index]
            img_id = self.imgs[index]
        elif self.test == True:
            index += self.split_factor
            caption = self.captions_test[index]
            img_id = self.imgs_test[index]
            
        img = Image.open(os.path.join(self.root_dir, img_id)).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        
        numericalized_caption = [self.vocab.stoi["<SOS>"]] #stoi is string to index, start of sentence
        numericalized_caption += self.vocab.numericalize(caption) # Convert each word to a number in our vocab
        numericalized_caption.append(self.vocab.stoi["<EOS>"])

        #return tensor
        
        return img, torch.tensor(numericalized_caption)
    
    @staticmethod
    def evaluation(self, index : int):
        caption = self.captions_test[index]
        img_id = self.imgs_test[index]
        img = Image.open(os.path.join(self.root_dir, img_id)).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)
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