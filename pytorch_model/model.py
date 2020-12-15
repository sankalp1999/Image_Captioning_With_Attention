import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class BahdanauAttention(nn.Module):
    
    '''
    Soft attention which is deterministic in nature. First introducted in 
    the paper Neural Machine Translation by Jointly Learning to Align and Translate (Bahdanau Et Al)

    '''
    
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        
        # Get the L attention dimension vector using this. Pass through softmax to get the 
        # score
        super(BahdanauAttention, self).__init__()
        
        self.attention = nn.Linear(attention_dim, 1)

        self.softmax = nn.Softmax(dim=1)
        
        self.relu = nn.ReLU()
        
        self.encoder_to_attention_dim = nn.Linear(encoder_dim, attention_dim)
        
        self.decoder_to_attention_dim = nn.Linear(decoder_dim, attention_dim)
        
        self.dropout = nn.Dropout(0.5)
        
        self.tanh = nn.Tanh()
        
    def forward(self, encoder_output, hidden_states):
        
        '''
        encoder_output : shape (batch_size, L, D)
        decoder_output : shape (batch_size, hidden_state dimension) 
        '''
      
        
        encoder_attention = self.encoder_to_attention_dim(encoder_output) # (batch_size, L, attention_dim)
        
        decoder_attention = self.decoder_to_attention_dim(hidden_states) # (batch_size, attention_dim)
        
        # Torch.cat() ?? 
        # >>> a = torch.cat((encoder,decoder.unsqueeze(1)),dim=1)
        # No, its actually adds the dim = 1 (Adds one more item in dim = 1)
        # We just want to add.
        
        
        #   (batch_size, L, attention_dim) + (batch_size, 1, attention_dim) 
        encoder_decoder = encoder_attention + decoder_attention.unsqueeze(1)  # (batch_size, L, attention_dim)
        
        encoder_decoder = self.tanh(encoder_decoder)
        
        attention_full = (self.attention(encoder_decoder)).squeeze(2) # (batch_size, L)
        
        alpha = self.softmax(attention_full) # Take the softmax across L(acc to paper)
        
        # One of the places where softmax is not used for classification only
        # but has a big role in life.(Mentioned in some lecture) Get a probablity distribution
        # where everything sums up to 1
        
        # Context vector is z
        # encoder_output is a aka annotation vector
        
        '''
        Equation 13 in the paper - classic Bahdanau attention
        '''
        
        z = (encoder_output * alpha.unsqueeze(2) ).sum(dim = 1) # Sum across L (pixels)
        
        return z, alpha

# Major changes include the ignoring of the last two layers. Author use a lower layer for more dense features.

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
        self.resnet50 = models.resnet50(pretrained=True)
        
        # Remove adaptive pool and FC from the end. 
        # Other working implementations leave only three but more features can be found
        # in the second last/third last layer
        layers_to_use = list(self.resnet50.children())[:-3]
        
        # Unpack and make it the conv_net
        self.resnet = nn.Sequential(*layers_to_use)
        
#         self.fc = nn.Linear(in_features,encoded_size)
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_size, encoded_size))
        
        self.relu = nn.ReLU()
        
        self.dropout = nn.Dropout(0.5)
        
        if not train_CNN:
            for param in self.resnet.parameters():
                param.requires_grad = False
           
        
    def forward(self, images):
         
        # images.shape (batch_size, 3, image_size, image_size)    
            
        # Change the image_size dimensions. Check them yourself.
        batch_size = images.shape[0]
        
        with torch.no_grad():
            features = self.resnet(images) # Shape : (batch_size, encoder_dim, image_size/32, image_size/32)
        
        features = self.adaptive_pool(features) # Shape (batch_size, encoder_dim, encoded_size, encoded_size)
        
        features = features.permute(0, 2, 3, 1) # Shape : (batch_size, encoded_size, encoded_size, encoder_dim)
        
        # The above transformation is needed because we are going to do some computation in the 
        # decoder.
        encoder_dim = features.shape[-1]
        # When in doubt https://stackoverflow.com/questions/42479902/how-does-the-view-method-work-in-pytorch
        features = features.view(batch_size, -1, encoder_dim)  # (batch_size, L, D)
        
        # print("-" * 80 )
        # print("Features shape : " , features.shape)
        # print("-" * 80 )
        
        return features
    
# In decoder, we use an LSTM cell. So, remove num_layers
# https://stackoverflow.com/questions/57048120/pytorch-lstm-vs-lstmcell
# In seq to seq model, it's more like gettign the state and ending the for loop when 
# you get the <EOS>

class Decoder(nn.Module):
    
    '''
    Get encoder output, pass through attention network, get attention weight and context_vector.
    Pass through the LSTMCell, predict.
    '''

    def __init__(self,encoder_dim, decoder_dim, embed_size, vocab_size, attention_dim, device, dropout = 0.5):
        
        super(Decoder,self).__init__()
        
        # Setting everything for the perfect model!
        
        self.device = device
        
        self.embed = nn.Embedding(vocab_size, embed_size) # Embedding layer courtesy Pytorch
        
        self.encoder_dim = encoder_dim
        
        self.decoder_dim = decoder_dim
        
        self.attention_dim = attention_dim
        
        self.embed_dim = embed_size
        
        self.vocab_size = vocab_size
        
        self.dropout = dropout
        
        # Note, it's an LSTM Cell, features + embedding
        self.lstm = nn.LSTMCell(self.encoder_dim + self.embed_dim, self.decoder_dim, bias=True)
        
        self.attention = BahdanauAttention(encoder_dim, decoder_dim, attention_dim)
        
        self.f_beta = nn.Linear(self.decoder_dim, self.encoder_dim)
        
        self.sigmoid = nn.Sigmoid()
        
        # See the paper 
        '''
        The initial memory state and hidden state of the LSTM
        are predicted by an average of the annotation vectors fed.
        through two separate MLPs (init,c and init,h):
        '''
        
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        
        # Not sure if I will use deep output layers
        # deep output layers
        self.L_h = nn.Linear(decoder_dim, embed_size, bias=False)
        self.L_z = nn.Linear(encoder_dim, embed_size, bias=False)
        self.L_o = nn.Linear(embed_size, vocab_size, bias=False)
        
        self.tanh = nn.Tanh()
        self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary
        self.init_weights()
        # self.init_weights()
        
    # Encoder output is the annotated vector a (L,D) in the paper
    
    def initialise_hidden_states(self, encoder_output):
        
        '''
        Initialise the hidden states before forward prop. As given in the paper.
        Authors take the mean of annotation vector across L dimension. Pass it through an MLP.
        '''
        # encoder_output : shape (batch_size, L, encoder_dim=D)
        
        mean = (encoder_output).mean(dim = 1) # Take mean over L
        
        # Pass through Fully connected
        
        c_0 = self.init_c(mean)
        c_0 = self.tanh(c_0)

        h_0 = self.init_h(mean)
        h_0 = self.tanh(h_0)
        
        return h_0, c_0, 

    def init_weights(self):
        
        # This helps initially. Fill the following weights before starting.
        
        self.embed.weight.data.uniform_(-0.1,0.1)
        self.fc.weight.data.uniform_(-0.1,0.1)
        self.fc.bias.data.fill_(0)
    
    # Thankful to sgrvinod implementation for this part. 
    # Note that without :batch_size_t i.e without using 
    # padded sequences, it is possible to 
    # train using the ignore_index in the loss function (CrossEntropy) 

    def forward(self, encoder_output, caption, caption_lengths):
        
        '''
        encoder_output : shape(batch_size, L, D)
        caption : (max_length, batch_size )
        
        Get the encoder_output i.e the features.
        '''
        
        device = self.device

        batch_size = encoder_output.size(0)
        # num_pixels 
        L = encoder_output.size(1)
        
        max_caption_length = caption.shape[-1] # shape : (batch_size, max_caption) 
        
        # Trick for fast training and avoiding <PAD> during forward of Decoder
        # Also, to use pack_padded_sequence(see train.py), we need sorted sequences
        # This helps to keep min padded elements at top and so on.
        caption_lengths, sort_ind = caption_lengths.sort(dim=0, descending=True)
        encoder_output = encoder_output[sort_ind]
        caption = caption[sort_ind]
        
        # It is possible to avoid this, not use pack_padded_sequences and evaluate loss
        # just by reshaping. But this method is more sophisticated, slightly faster and 
        # it has been more used in the available implementations.
     
        # print(sort_ind)
        
        # We won't decode at <EOS> i.e the last time step
        lengths = [l - 1 for l in caption_lengths]
        
        embedding_of_all_captions = self.embed(caption)
        
        predictions = torch.zeros(batch_size, max_caption_length - 1, self.vocab_size).to(device)
        alphas = torch.zeros(batch_size, max_caption_length - 1, L).to(device)  
        
        # Concat and pass through lstm to get hidden states
        
        h, c = self.initialise_hidden_states(encoder_output)
        
        # Exclude <EOS>, t is the th timestep
        # We get all the embeddings for the t timestep
        # Then we get the encoded_output aka annotation vector
        # Use soft attention to get the context vector.
        # Concat and pass through the lstm cell to get hidden states --> predictions
        
        # print(max_caption_length)
        
        for t in range(max_caption_length - 1):
            
            batch_size_t = sum([l > t for l in lengths]) 
            
            # z from the returning function
            context_vector, alpha = self.attention(encoder_output[:batch_size_t], h[:batch_size_t])
            
            # Changes inspirsed from SgdrVinod(Suggested in paper also)
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))
            
            gated_context = gate * context_vector
            # context_vector : torch.Size([32, 1024]), embedded_caption_t : torch.Size([32, 256])

            h, c = self.lstm(torch.cat([ embedding_of_all_captions[:batch_size_t,t,:], gated_context], dim=1),(h[:batch_size_t], c[:batch_size_t]))
            
            predict_deep = self.deep_output_layer(embedding_of_all_captions[:batch_size_t,t,:], h, context_vector)
            
            predictions[:batch_size_t, t, :] = predict_deep 
            
            alphas[:batch_size_t, t, :] = alpha
            
        return predictions, alphas, caption, lengths
        
    
    def deep_output_layer(self, embedded_caption, h, context_vector):
        """
        :param embedded_caption: embedded caption, a tensor with shape (batch_size, embed_dim)
        :param h: hidden state, a tensor with shape (batch_size, decoder_dim
        :param context_vector: context vector, a tensor with shape (batch_size, encoder_dim)
        :return: output
        """
        
        # Not working properly in early part of training. I don't know about this clearly.
        # scores = self.L_o(self.dropout(embedded_caption + self.L_h(h) + self.L_z(context_vector)))
        
        ## UPDATE: Check Flickr30K model changes for a 2 layer neural network(Deep output RNN). 
        
        dropout = nn.Dropout(0.2)
        scores = dropout(self.fc(h))
        return scores

    # Greedy. Just keep passing the prediction and select the one with the top score.
    def predict_caption(self, encoder_output, captions):
        
        # "<SOS>" 1
        caption_list = [1]
        alphas = [] 
        h, c = self.initialise_hidden_states(encoder_output)
        
        
        # 2 is <EOS>
        while len(caption_list) < 40 :
            word = caption_list[-1]
            
            embedded_caption = self.embed(  torch.LongTensor([word]).to(self.device)  )  # (1, embed_dim)
            
            context_vector, alpha = self.attention(encoder_output, h)
            
            gate = self.sigmoid(self.f_beta(h))
            
            gated_context = gate * context_vector
            
            h, c = self.lstm(torch.cat([embedded_caption, gated_context], dim=1), (h, c))
            
            predictions = self.deep_output_layer(embedded_caption, h, context_vector)  # (1, vocab_size)
            
            # item converts to python scalar otherwise expect CUDA re-assert trigger
            
            next_word = (torch.argmax(predictions, dim=1, keepdim=True).squeeze()).item()
            
            caption_list.append(next_word)
            
            alphas.append(alpha)
            
            if(caption_list[-1] == 2):
                break
        return caption_list, alphas


    # At each time step, keep track of top k predictions and scores. This can be easily understood
    # if you know the understanding of a heap(Pretty unrelated here)
    # It turns out that we need to do this since all words are dependent 
    # on the word your model predicted earlier. If you do it greedily (Beam_index = 1),
    # then if you are first choice is sub-optimal, then the whole sequence will be sub-optimal.
    
    # This feels like a dynamic programming problem in the sense that you 
    # need to try several combinations.

    # So keep track of k sequences and choosing the sequence with max score proves to better
    # in most cases. Refer https://www.youtube.com/watch?v=RLWuzLLSIgw  

    # Inspired from Sgrvinod implementation(almost same)
    # This is the game changer in many scenarios. 
    def beam_search(self, encoder_output, beam_size = 3):
        
        device = self.device

        k = beam_size
        
        vocab_size = self.vocab_size
        
        encoder_size = encoder_output.size(-1)
        
        encoder_output = encoder_output.view(1, -1, encoder_size)
        
        num_pixels = encoder_output.size(1)
        
        # Get the annotated vector(Features) and repeat across k dimensions
        encoder_output = encoder_output.expand(k, num_pixels, encoder_size)  # (k, num_pixels, encoder_dim)
        
        # Vocab.stoi(SOS)
        k_prev_words = torch.LongTensor([[1]] * k).to(device) 
        seqs = k_prev_words
        
        # To store the scores at each stage
        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)
        
        # Store sequences and scores
        complete_seqs = list()
        complete_seqs_scores = list()
        
        step = 1
        
        
        h, c = self.initialise_hidden_states(encoder_output)
        
        while True:

            # Get embedding
            embedded_caption = self.embed(k_prev_words).squeeze(1)
            
            # Pass through the attention network
            context_vector, alpha = self.attention(encoder_output, h);
            
            gate = self.sigmoid(self.f_beta(h))
            
            gated_context = gate * context_vector
            
            h, c = self.lstm(torch.cat([embedded_caption, gated_context], dim=1), (h, c))
            
            # Used normal Fully connected layer instead with little dropout.
            scores = self.deep_output_layer(embedded_caption, h, context_vector)
            
            scores = F.log_softmax(scores, dim=1)
            
            scores = top_k_scores.expand_as(scores) + scores
            
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, dim=0)  # (s)
            else:
                top_k_scores, top_k_words = scores.view(-1).topk(k, dim=0)  # (s)
                
            prev_word_inds = torch.true_divide(top_k_words , vocab_size).long().cpu()  # (s)
            next_word_inds = top_k_words % vocab_size  # (s)
            
             # Add new words to sequences
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

            # Which sequences are incomplete (didn't reach <EOS>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != 2 ] #vocab.itos['<EOS>']
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            
            k -= len(complete_inds)  # reduce beam length accordingly

            # Proceed with incomplete sequences
            if k == 0:
                break
                
            seqs = seqs[incomplete_inds]
            
            h = h[prev_word_inds[incomplete_inds]]
            c = c[prev_word_inds[incomplete_inds]]
            
            encoder_output = encoder_output[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            # Can't detect
            if step > 50:
                break
            step += 1

             
        if len(complete_seqs_scores) > 0:
            i = complete_seqs_scores.index(max(complete_seqs_scores))
            seq = complete_seqs[i]
            return seq
        else:
            return [1,2] # If cannot predict <EOS> in the beginning of training because I was checking an untrained network, very dumb of me.
        return complete_seqs


    
