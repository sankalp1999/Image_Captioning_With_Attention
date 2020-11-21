
from torchtext.data.metrics import bleu_score
import nltk

# Please don't judge me by this file.
# I don't know if this is fully correct but please contact on GitHub 
# or personally if you can to tell me the correct procedure.


def bleu_score_checker():
    '''
    This is proper BLEU score checker which checks a certain number of images.
    These are 500 images (2500 captions) that were never shown to the model.

    You can see some of the outputs in a folder or README.

    '''
    # Corpus files
    gc = []
    test = []
    # Will execute only 200 times. Inner 5 times
   
    offset = 37500
    for i in range(250,500,5):
        temp_gc = []
        encoder.eval() # Very important to switch off dropout and batchnorm
        decoder.eval()
        with torch.no_grad():
            img, caption = dataset.evaluation(dataset,i + offset)
            img = img.unsqueeze(0)
#             print(caption)
            
            encoded_output = encoder(img.to(device))
            
            caps = decoder.beam_search(encoded_output, 4)
            
            caps = [dataset.vocab.itos[idx] for idx in caps]
            
            generated_caption = ' '.join(caps)
            # show_image2(img.squeeze(0),i,title=generated_caption)
            
            caption = caption.split()[:-1]
            generated_caption = generated_caption.split()[1:]
            generated_caption = generated_caption[:-2]
            
            test.append(generated_caption)
            temp_gc.append(caption)
        for j in range(1,5):
            img, caption = dataset.evaluation(dataset, i + j + offset)
            temp_gc.append(caption)
        gc.append(temp_gc)
        decoder.train()
    
    print("-" * 80)

    print("Torch metrics")
    
    print("BLEU-1", bleu_score(test,gc, max_n = 1, weights = [1] ) )
    
    print("BLEU-2", bleu_score(test, gc, max_n = 2, weights = [0.5,0.5]))
    
    print("BLEU-3", bleu_score(test, gc, max_n = 3, weights = [0.33,0.33,0.33]))
    
    print("BLEU-4", bleu_score(test, gc, max_n = 4, weights = [0.25,0.25, 0.25, 0.25]))
    
    
    print("-"*80)
    print("Nltk metrics")
    BLEU4 = nltk.translate.bleu_score.corpus_bleu(gc, test,weights=(0.25,0.25,0.25,0.25))
    BLEU1 = nltk.translate.bleu_score.corpus_bleu(gc, test,weights=(1.0,0,0,0))
    BLEU2 = nltk.translate.bleu_score.corpus_bleu(gc, test,weights=(0.5,0.5,0,0))
    BLEU3 = nltk.translate.bleu_score.corpus_bleu(gc, test,weights=(0.33,0.33,0.33,0))
    
    
    print(f"BLEU-1 {BLEU1}")
    print(f"BLEU-2 {BLEU2}")
    print(f"BLEU-3 {BLEU3}")
    print(f"BLEU-4 {BLEU4}")
    
    print("GC" , gc)
    print("Predictions", test)


def checker():
    '''
    Check the of caption predictions on train images to see
    how model performs and what changes to make.

    '''
    
    gc = []
    test = []
    # Will execute only 200 times. Inner 5 times
    offset = 0
    for i in range(100,200,5):
        temp_gc = []
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            img_id, caption =  (dataset.imgs[i]), dataset.captions[i]
            img = Image.open(os.path.join(dataset.root_dir, img_id)).convert("RGB")
            img = dataset.transform(img)
            img = img.unsqueeze(0)
            # print(caption)
            
            encoded_output = encoder(img.to(device))
            for k in range(1,5):
                caps = decoder.beam_search(encoded_output, k)

                caps = [dataset.vocab.itos[idx] for idx in caps]

                generated_caption = ' '.join(caps)
                generated_caption = generated_caption.split()[1:]
                generated_caption = generated_caption[:-2]

                print(generated_caption)

            test.append(generated_caption)

            show_image2(img.squeeze(0),i, title=' '.join(generated_caption) )
            caption = caption.split()[:-1]
            
            print("Original", caption)
            temp_gc.append(caption)
        for j in range(1,5):
            img, caption = (dataset.imgs[i + j]), dataset.captions[i + j]
            
            temp_gc.append(caption)
        gc.append(temp_gc)
        decoder.train()
        encoder.train()
    
    print("-" * 80) # Learnt this from @SahilKhose
    
    print("Torch metrics")
    
    print("BLEU-1", bleu_score(test,gc, max_n = 1, weights = [1] ) )
    
    print("BLEU-2", bleu_score(test, gc, max_n = 2, weights = [0.5,0.5]))
    
    print("BLEU-3", bleu_score(test, gc, max_n = 3, weights = [0.33,0.33,0.33]))
    
    print("BLEU-4", bleu_score(test, gc, max_n = 4, weights = [0.25,0.25, 0.25, 0.25]))
    
    
    print("-"*80)
    print("Nltk metrics")
    BLEU4 = nltk.translate.bleu_score.corpus_bleu(gc, test,weights=(0.25,0.25,0.25,0.25))
    BLEU1 = nltk.translate.bleu_score.corpus_bleu(gc, test,weights=(1.0,0,0,0))
    BLEU2 = nltk.translate.bleu_score.corpus_bleu(gc, test,weights=(0.5,0.5,0,0))
    BLEU3 = nltk.translate.bleu_score.corpus_bleu(gc, test,weights=(0.33,0.33,0.33,0))
    
    
    print(f"BLEU-1 {BLEU1}")
    print(f"BLEU-2 {BLEU2}")
    print(f"BLEU-3 {BLEU3}")
    print(f"BLEU-4 {BLEU4}")
    
    # To see the sentences
    # print("GC" , gc)
    # print("Predictions", test)
        

