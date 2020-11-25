import pickle
from model import BahdanauAttention, EncoderCNN, Decoder
from vocab import Vocab_Builder
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image, ImageOps
import io
import time
import streamlit as st
import requests
import os
from io import BytesIO
import wget

device = 'cpu'

st.set_page_config(
    initial_sidebar_state="expanded",
    page_title="CaptionBot"
)


def transform_image(image):
    mean = [0.485, 0.456, 0.406]

    std = [0.229, 0.224, 0.225]

    transform = transforms.Compose(
        [transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)]
    )
#     image = Image.open(io.BytesIO(img_bytes) ).convert("RGB")
    return transform(image)

def load_checkpoint(checkpoint, model, optimizer):
    
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    step = checkpoint["step"]
    return step

class EncoderCNN(nn.Module):
   
    '''
    Takes in the image, encode it in shape (L,D) and return to decoder
        
     "The extractor produces L vectors, each of which is
    a D-dimensional representation corresponding to a part of
     the image"
        
    '''
#     @st.cache(ttl=3600,max_entries=10) solved internal hash error
    def __init__(self, encoded_size=14, train_CNN = False):
        
        super(EncoderCNN, self).__init__()
        
        # Fine-tune parameter
        self.train_CNN = train_CNN
        
        self.encoded_size =encoded_size
        
        # Load the resnet, but pretrained = False if you want to just load the weights
        self.resnet50 = models.resnet50(pretrained=False)
        # Remove adaptive pool and FC from the end. 
        # Other working implementations leave only three but more features can be found
        # in the second last/third last layer
        layers_to_use = list(self.resnet50.children())[:-3]
        
        # Unpack and make it the conv_net
        self.resnet = nn.Sequential(*layers_to_use)
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_size, encoded_size))
        
        self.relu = nn.ReLU()
        
        self.dropout = nn.Dropout(0.5)
        
        if not train_CNN:
            for param in self.resnet.parameters():
                param.requires_grad = False
        
    def forward(self, images):
         
        # images.shape (batch_size, 3, image_size, image_size)    
            
        batch_size = images.shape[0]
        
        with torch.no_grad():
            features = self.resnet(images)              
        features = self.adaptive_pool(features) 
        features = features.permute(0, 2, 3, 1) 
        
        # The above transformation is needed because we are going to do some computation in the 
        # decoder.
        encoder_dim = features.shape[-1]
        # When in doubt https://stackoverflow.com/questions/42479902/how-does-the-view-method-work-in-pytorch
        features = features.view(batch_size, -1, encoder_dim)  # (batch_size, L, D)

        return features

@st.cache
def download_data():
    
    path1 = './LastModelResnet50_v2_16.pth.tar'
    path2 = './resnet50_captioning.pt'
    
    # Local
    # path1 = './data/LastModelResnet50_v2_16.pth.tar'
    # path2 = './data/resnet50_captioning.pt'
    # print("I am here.")
    
    if not os.path.exists(path1):
        decoder_url = 'wget -O ./LastModelResnet50_v2_16.pth.tar https://www.dropbox.com/s/5ntq1bgp33k1197/LastModelResnet50_v2_16.pth.tar?dl=0'
        
        with st.spinner('done!\nmodel weights were not found, downloading them...'):
            os.system(decoder_url)
    else:
        print("Model 1 is here.")

    if not os.path.exists(path2):
        encoder_url = 'wget -O ./resnet50_captioning.pt https://www.dropbox.com/s/fot9zzgszkpsab7/resnet50_captioning.pt?dl=0'
        with st.spinner('Downloading model weights for resnet50'):
            os.system(encoder_url)
    else:
        print("Model 2 is here.")

@st.cache
def load_model():
    
    # global vocab
    vocab = Vocab_Builder(freq_threshold = 2)

    # Load the pickle dump
    vocab_path = './vocab.pickle'

    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    print(len(vocab))
    embed_size = 256
    encoder_dim = 1024
    decoder_dim = 400
    attention_dim = 400
    vocab_size = len(vocab)
    learning_rate = 2e-4 
    resnet_path = './resnet50_captioning.pt'
    
    # global encoder
    encoder = EncoderCNN()

    # Don't want to download pretrained resnet again even though not even fine-tuned!
    encoder.load_state_dict( torch.load( resnet_path, map_location = 'cpu') )
    encoder.to(device)
    encoder.eval() # V. important to switch off Dropout and BatchNorm

    decoder_path = './LastModelResnet50_v2_16.pth.tar'

    # global decoder
    decoder = Decoder(encoder_dim, decoder_dim, embed_size, vocab_size, attention_dim, device)    

    optimizer = optim.Adam(decoder.parameters(), lr = learning_rate)
    
    checkpoint = torch.load(decoder_path,map_location='cpu')
    decoder.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    step = checkpoint["step"]

    # return step
    #   step = load_checkpoint(torch.load(decoder_path ,map_location = 'cpu'), decoder, optimizer)

    decoder = decoder.to(device)
    decoder.eval()
    
    return vocab, encoder, decoder

def predict_caption(image_bytes):
    
    captions = []
    img_t = transform_image(image_bytes)
    for i in range(3,7):
        encoded_output = encoder(img_t.unsqueeze(0).to(device))
        caps = decoder.beam_search(encoded_output,i)
        caps = caps[1:-1]
        caption = [vocab.itos[idx] for idx in caps]
        caption = ' '.join(caption)
        print(caption)
        captions.append(caption)
    for i in range(len(captions)):
        s = ("** Prediction " + str(i + 1) + ": " + captions[i] + "**")
        st.markdown(s)        

@st.cache(ttl=3600, max_entries=10)
def load_output_image(img):
    
    if isinstance(img, str): 
        image = Image.open(img)
    else:
        img_bytes = img.read() 
        image = Image.open(io.BytesIO(img_bytes) ).convert("RGB")
    
    # Auto - orient refer https://stackoverflow.com/a/58116860
    image = ImageOps.exif_transpose(image) 
    return image

@st.cache(ttl=3600, max_entries=10)
def pypng():
    image = Image.open('data/pytorch.png')
    return image
    
if __name__ == '__main__':

    download_data()
    vocab, encoder, decoder = load_model()
    
    pytorch_image = pypng()
    st.image(pytorch_image, width = 500)
    
    st.title("The Image Captioning Bot")
    st.text("")
    st.text("")
    st.success("Welcome! Please upload an image!"
    )   

    st.info("If nothing happens after 10 seconds of uploading, reload the page and select again.")
    
    args = { 'sunset' : 'imgs/sunset.jpeg' }
    
    img_upload  = st.file_uploader(label= 'Upload Image', type = ['png', 'jpg', 'jpeg'])
    
    img_open = args['sunset'] if img_upload is None else img_upload
    
    image = load_output_image(img_open)
    
#     st.sidebar.title("Tips")
    st.sidebar.markdown('''
    # Pro Tips
    If you are getting funny predictions \n
    1. Prefer using the app from PC
    2. Upload less complex images.
    3. CaptionBot likes dogs, men, women and kids.
    4. Profile pictures(Whatsapp) are \n good candidates!
    ''')
    
    st.sidebar.markdown('''Check the source code [here](https://github.com/sankalp1999/Image_Captioning)
    \n Liked it? Give a :star:  on GitHub ''')
    
    st.image(image,use_column_width=True,caption="Your image")

    # img_bytes earlier
    if st.button('Generate captions!'):
        predict_caption(image)
        st.success("Click again to retry or try a different image by uploading")
        st.balloons()
