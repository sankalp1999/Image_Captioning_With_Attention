import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
# from convert_text import get_loader
from utils import *
from model import *
from dataset import get_loader


data_location =  'flickr8k'
caption_file = 'flickr8k/captions.txt'

mean = [0.485, 0.456, 0.406]

std = [0.229, 0.224, 0.225]

transform = transforms.Compose(
    [transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)]
)

train_loader, dataset = get_loader(
    root_folder = data_location+"/Images",
    annotation_file = data_location+"/captions.txt",
    transform = transform, 
    batch_size = 1,
    num_workers = 4,
    test = False
)
test_loader, test_dataset = get_loader(
    root_folder = data_location+"/Images",
    annotation_file = data_location+"/captions.txt",
    transform = transform, 
    num_workers = 4,
    test = True
)


# Test_dataset gonna come here soon
# Think about that later. We will do some training phases. It will take time but keep your calm.

torch.backends.cudnn.benchmark = True # Get some boost probaby

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
print(device)

load_model = False

save_model = False

train_CNN = False


# Hyperparameters

embed_size = 256
encoder_dim = 1024
decoder_dim = 400
attention_dim = 400
vocab_size = len(dataset.vocab)
learning_rate = 2e-4 # Earlier 3e-4
num_epochs = 21

# Tensorboard
writer = SummaryWriter("runs/flickr")

step = 0
# init model, loss
encoder = EncoderCNN() # Default arguments already given as encoder_size 14, train_CNN = False
encoder = encoder.to(device)

decoder = Decoder(encoder_dim, decoder_dim, embed_size, vocab_size, attention_dim, device)    
decoder = decoder.to(device)

alpha_c = 1.0 
# regularization parameter for 'doubly stochastic attention', as in the paper
# https://pytorch.org/tutorials/beginner/torchtext_translation_tutorial.html
# criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"]).to(device)

criterion = nn.CrossEntropyLoss().to(device)


# Optimizer only for the decoder
optimizer = optim.Adam(decoder.parameters(), lr = learning_rate)

decoder.train()
if load_model:
    step = load_checkpoint(torch.load("../input/resnet15/LastModelResnet50_v2_16.pth.tar"), decoder, optimizer)

    
# Change the epoch before changing    
for epoch in range(0, num_epochs):
    if save_model:
        checkpoint = {
            "state_dict" : decoder.state_dict(),
            "optimizer" : optimizer.state_dict(),
            "step" : step
        }

        if epoch % 5 == 0 and epoch != 0 or epoch > 15:
            filename = './LastModelResnet50_v2_' + str(epoch) +  '.pth.tar'
            save_checkpoint(checkpoint, filename)
    losses = []
    mvl = []
    for idx, (imgs, captions, lengths) in enumerate(train_loader):
        # optimizer.zero_grad() Init config

        imgs = imgs.to(device)
        captions = captions.to(device)
        lengths = lengths.to(device)
        # Pass through the encoder and get the annotation vector
        encoded_images = encoder(imgs)

        scores, alphas, sorted_cap, decode_lengths = decoder(encoded_images, captions, lengths)
        
        # We don't want <SOS>
        sorted_cap = sorted_cap[:,1:] # shape (batch_size, max_caption)
        
        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
        targets = pack_padded_sequence(sorted_cap, decode_lengths, batch_first=True).data
        
        # Calculate loss
        loss = criterion(scores, targets)
        
        # This method also works if you use ignore_index
        # batch_size = sorted_cap.shape[0]
        # caption_length = sorted_cap.shape[1]
        # loss = criterion(predictions.view(batch_size * caption_length, -1), sorted_cap.reshape(-1))

        #   Doubly stochastic attention regularization 
        loss += 1.0 * ((1. - alphas.sum(dim=1)) ** 2).mean()

        losses.append(loss.item())
        
        decoder.zero_grad()
        encoder.zero_grad()
        
        loss.backward()

        print("Step", idx, loss.item())

        writer.add_scalar("training loss", loss.item(), global_step = step)

        step += 1

        optimizer.step()
        
        # This part is mostly manual for checking.
        
        if (idx + 1 )% 100 == 0:
            print("Epoch: {} loss: {:.5f}".format(epoch,loss.item()))

            decoder.eval()
            with torch.no_grad():
                # bleu_score_checker()
                dataiter = iter(train_loader)
                imgs,captions, lengths = next(dataiter)
                imgs = imgs
                captions = captions.to(device)
                encoded_output = encoder( (imgs[0].unsqueeze(0).to(device)) )

                # Does not make a difference for the caption since we are not using it
                caption_greedy, alphas = decoder.predict_caption(encoded_output, captions)
                caps_greedy =[dataset.vocab.itos[idx] for idx in caption_greedy]
                caption = ' '.join(caps_greedy)
                print("Greedy search", caption)
                
                caption = decoder.beam_search(encoded_output, 4)
                caps = [dataset.vocab.itos[idx] for idx in caption]
                print("Beam search", ' '.join(caps) )

                show_image(imgs[0],title=' '.join(caps))
            decoder.train()
            encoder.train() # Messed up while copying from kaggle kernel
                            # We are not training the encoder but eval switches off 
                            # Batch norm and dropout
        
        # Valid loss without another dataloader. I created a manual split of images and captions
        if (idx + 1 ) % 200 == 0 :

            valid_losses = []
            
            decoder.eval() # V. Important.
            encoder.eval()

            print("Valid section")
            with torch.no_grad():
                for index, (imgs, captions, lengths) in enumerate(test_loader):
                    
                    imgs = imgs.to(device)
                    captions = captions.to(device)
                    lengths = lengths.to(device)
                    
                    encoded_images = encoder(imgs)
                    scores, alphas, sorted_cap, decode_lengths = decoder(encoded_images, captions, lengths)

                    # We don't want <SOS>
                    sorted_cap = sorted_cap[:,1:] # shape (batch_size, max_caption)

                    scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
                    targets = pack_padded_sequence(sorted_cap, decode_lengths, batch_first=True).data

                    valid_loss = criterion(scores, targets)
                
                    # This also works if you use ignore_index.
                    # batch_size = sorted_cap.shape[0]
                    # caption_length = sorted_cap.shape[1]
                    #  valid_loss = criterion(predictions.view(batch_size * caption_length, -1), sorted_cap.reshape(-1))
                    
                    valid_loss += 1.0 * ((1. - alphas.sum(dim=1)) ** 2).mean()
                    
                    valid_losses.append(valid_loss.item())

                    print("Step", index, valid_loss.item())
                    
            decoder.train()
            encoder.train()
            print("-" * 80)
            
            mean_valid_loss = sum(valid_losses)/len(valid_losses)
            
            mvl.append(mean_valid_loss)
            print(mean_valid_loss)        
            
            print("-" * 80)                        
    
    mean_loss = sum(losses)/len(losses)
    print("Mean loss", mean_loss)
    print(mvl)


