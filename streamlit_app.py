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
    page_title="CaptionBot 2.0"
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


@st.cache
def download_data():
    
    path1 = './Flickr30k_Decoder_10.pth.tar'
    path2 = './resnet5010.pt'
    
    # Local
    # path1 = './data/LastModelResnet50_v2_16.pth.tar'
    # path2 = './data/resnet50_captioning.pt'
    # print("I am here.")
    
    if not os.path.exists(path1):
        decoder_url = 'wget -O ./Flickr30k_Decoder_10.pth.tar https://www.dropbox.com/s/cf2ox65vi7c2fou/Flickr30k_Decoder_10.pth.tar?dl=0'
        
        with st.spinner('done!\nmodel weights were not found, downloading them...'):
            os.system(decoder_url)
    else:
        print("Model 1 is here.")

    if not os.path.exists(path2):
        encoder_url = 'wget -O ./resnet5010.pt https://www.dropbox.com/s/v0ikcdbh8w2rqii/resnet5010.pt?dl=0'
        with st.spinner('Downloading model weights for resnet50'):
            os.system(encoder_url)
    else:
        print("Model 2 is here.")

@st.cache
def load_model():
    
    # global vocab
    vocab = Vocab_Builder(freq_threshold = 5)

    # Load the pickle dump
    vocab_path = './vocab (1).pickle'

    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    print(len(vocab))
    embed_size = 350
    encoder_dim = 1024
    decoder_dim = 512
    attention_dim = 512
    vocab_size = len(vocab)
    learning_rate = 4e-5 # Modifed it after 10th epoch
    # resnet_path = './resnet50_captioning.pt'
    resnet_path = './resnet5010.pt'
    # global encoder
    encoder = EncoderCNN()

    # Don't want to download pretrained resnet again even though not even fine-tuned!
    encoder.load_state_dict( torch.load( resnet_path, map_location = 'cpu') )
    encoder.to(device)
    encoder.eval() # V. important to switch off Dropout and BatchNorm

    # decoder_path = './LastModelResnet50_v2_16.pth.tar'
    decoder_path = './Flickr30k_Decoder_10.pth.tar'
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
    for i in range(1,6):
        encoded_output = encoder(img_t.unsqueeze(0).to(device))
        caps = decoder.beam_search(encoded_output,i)
        caps = caps[1:-1]
        caption = [vocab.itos[idx] for idx in caps]
        caption = ' '.join(caption)
        print(caption)
        captions.append(caption)
    for i in range(len(captions)):
        s = ("** Beam index " + str(i + 1) + ": " + captions[i] + "**")
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

    
    args = { 'sunset' : 'imgs/sunset.jpeg' }
    
    img_upload  = st.file_uploader(label= 'Upload Image', type = ['png', 'jpg', 'jpeg'])
    
    img_open = args['sunset'] if img_upload is None else img_upload
    
    image = load_output_image(img_open)
    
#     st.sidebar.title("Tips")
    st.sidebar.markdown('''
    # Pro Tips
    If you are getting funny predictions \n
    1. Prefer using the app from PC :computer:
    2. Upload less complex image.
    3. CaptionBot likes dogs :dog: , men, women and kids. Sorry catlovers.
    4. Profile pictures(Whatsapp) are \n good candidates!
    
    **Try this** :wink:

    If greater than/equal to two captions say
    you are woman, then you are more
    feminine looking and vice-versa.
    Upload a close-up to see! 
    
    ''')
    
    st.sidebar.markdown('''Check the source code [here](https://github.com/sankalp1999/Image_Captioning)
    \n Liked it? Give a :star:  on GitHub ''')
    
    st.image(image,use_column_width=True,caption="Your image")

    # img_bytes earlier
    if st.button('Generate captions!'):
        predict_caption(image)
        st.success("Click again to retry or try a different image by uploading")
        st.balloons()
 
