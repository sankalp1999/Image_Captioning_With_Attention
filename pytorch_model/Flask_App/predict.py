import pickle
from model import BahdanauAttention, EncoderCNN, Decoder
from vocab import Vocab_Builder
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import io

device = 'cpu'

def transform_image(path):
    mean = [0.485, 0.456, 0.406]

    std = [0.229, 0.224, 0.225]

    transform = transforms.Compose(
        [transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)]
    )
    image = Image.open(path).convert("RGB")
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



def load_model():
    global vocab
    vocab = Vocab_Builder(freq_threshold = 2)

    # Load the pickle dump
    vocab_path = '/home/sankalp/Desktop/Image_Caption_With_Attention/vocab.pickle'
    

    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    

    print(len(vocab))
    embed_size = 256
    encoder_dim = 1024
    decoder_dim = 400
    attention_dim = 400
    vocab_size = len(vocab)
    learning_rate = 2e-4 



    resnet_path = '/home/sankalp/Desktop/Image_Caption_With_Attention/resnet50_captioning.pt'
    global encoder
    encoder = EncoderCNN()

    # Don't want to download pretrained resnet again even though not even fine-tuned!
    encoder.load_state_dict( torch.load(resnet_path, map_location = 'cpu') )
    encoder.to(device)

    encoder.eval() # V. important to switch off Dropout and BatchNorm

    decoder_path = '/home/sankalp/Desktop/Image_Caption_With_Attention/LastModelResnet50_v2_16.pth.tar'

    global decoder
    decoder = Decoder(encoder_dim, decoder_dim, embed_size, vocab_size, attention_dim, device)    


    optimizer = optim.Adam(decoder.parameters(), lr = learning_rate)

    step = load_checkpoint(torch.load(decoder_path ,map_location = 'cpu'), decoder, optimizer)

    decoder = decoder.to(device)
    decoder.eval()


# image_path = 'flickr8k/Images/54501196_a9ac9d66f2.jpg'

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
    return captions










