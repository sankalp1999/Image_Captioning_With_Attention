import os
import pandas as pd
import spacy # for tokenizer
import torch
import pickle
# python -m spacy download en
spacy_eng = spacy.load("en")

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

