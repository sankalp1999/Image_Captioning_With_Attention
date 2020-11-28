
from torchtext.data.metrics import bleu_score
import nltk

# This is a proper BLEU score evaluator.

def bleu_score_checker():
    
    gc = []
    test = []
    # Will execute only 200 times. Inner 5 times
    offset = 37500
    
    # Go over all the test images ~500.
    for i in range(0,2900,5):
        temp_gc = []
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            img, caption = dataset.evaluation(dataset,i + offset)
            img = img.unsqueeze(0)
            # print(caption)
            encoded_output = encoder(img.to(device))
            caps = decoder.beam_search(encoded_output, 3)
            caps = [dataset.vocab.itos[idx] for idx in caps]
            generated_caption = ' '.join(caps)
            # show_image2(img.squeeze(0),i,title=generated_caption)
            generated_caption = generated_caption.split()[1:]
            generated_caption = generated_caption[:-2]
            test.append(generated_caption)
            temp_gc.append(caption)
        for j in range(1,5):
            img, caption = dataset.evaluation(dataset, i + j + offset)
            temp_gc.append(caption)
        gc.append(temp_gc)
        decoder.train()
        encoder.train()
    print("-" * 80)
    print("Torch metrics")
    print("BLEU-1", bleu_score(test,gc, max_n = 1, weights = [1.0] ) )
    
    print("BLEU-2", bleu_score(test, gc, max_n = 2, weights = [0.5,0.5]))
    
    print("BLEU-3", bleu_score(test, gc, max_n = 3, weights = [1/3,1/3,1/3]))
    
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
    
#     print("GC" , gc)
#     print("Predictions", test)
        
    


