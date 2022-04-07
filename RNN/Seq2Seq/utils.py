import spacy
import torch


def ger2en(model, sentence, german, english, device, max_length=50):
    spacy_ger = spacy.load('de_core_news_sm')
    
    if type(sentence) == str:
        tokens = [token.text.lower() for token in spacy_ger(sentence)]
    else:
        tokens = [token.lower() for token in sentence]
        
    tokens.insert(0, german.init_token)
    tokens.append(german.eos_token)
    
    tensor = torch.LongTensor([german.vocab.stoi[token] for token in tokens]).unsqueeze(1).to(device)
    
    with torch.no_grad():
        hidden, cell = model.encoder(tensor)
        
    outputs = [english.vocab.stoi[english.init_token]]
    
    for _ in range(max_length):
        prev = torch.LongTensor([outputs[-1]]).to(device)
        
        with torch.no_grad():
            output, hidden, cell = model.decoder(prev, hidden, cell)
            best = output.argmax(1).item()
        outputs.append(best)
        if output.argmax(1).item() == english.vocab.stoi[english.eos_token]:
            break
    
    translated = [english.vocab.itos[idx] for idx in outputs]
    return translated