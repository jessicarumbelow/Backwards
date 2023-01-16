import subprocess
import sys
import os
import torch

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    from transformers import GPT2Tokenizer, GPT2LMHeadModel, utils, AutoTokenizer, AutoModelForCausalLM
except:
    install('transformers')
    from transformers import GPT2Tokenizer, GPT2LMHeadModel, utils, AutoTokenizer, AutoModelForCausalLM

utils.logging.set_verbosity_error()

def load_all(model_name="gpt2", device='cpu'):
    cur_dir = os.listdir()
    
    if model_name + '_tokenizer' in str(cur_dir):
        print('Loading tokenizer...')
        tokenizer = torch.load(model_name + '_tokenizer')
    else:
        # Will have to change this line to support other models automatically
        print('Downloading tokenizer...')
        if 'gpt-j' in model_name:
            tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
        else:
            tokenizer = GPT2Tokenizer.from_pretrained(model_name, padding_side='left')
        torch.save(tokenizer, model_name + '_tokenizer')
    pad_token_id=tokenizer.eos_token_id


    if model_name + '_model' in str(cur_dir):
        print('Loading model...')
        model = torch.load(model_name + '_model').to(device)
    else:
        print('Downloading model...')

        if 'gpt-j' in model_name:
            model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B").to(device)
        else:
            model = GPT2LMHeadModel.from_pretrained(model_name, pad_token_id=tokenizer.eos_token_id, vocab_size=vocab_len).to(device)
        torch.save(model, model_name + '_model')
    model.eval()

    # 'word_embeddings' tensor gives emeddings for each token in the vocab for this model,
    # has shape (vocab_len, embedding_dimension) which in this case = (50257, 768)
    
    embeddings = model.transformer.wte.weight.to(device)
    if model_name + '_embeddings' not in str(cur_dir):
        torch.save(embeddings, model_name + '_embeddings')

    return model, embeddings, tokenizer




def normalise(x, min_max=[]):     
# normalises values of (array or tensor) x according to first (min) and second (max) values in list min_max. 
# This effectively defaults to [0,1] if the list doesn't contain exactly two elements. 
# The original code threw an error if min_max had length 1, so it's been changed slightly.

# First normalise x to [0,1]
    rnge = x.max() - x.min()
    if rnge > 0:
        x = (x - x.min())/rnge

# Now, if there's a min and max given in min_max list, multiply by difference and add minimum
    if len(min_max) > 1:
        rnge = min_max[1] - min_max[0]
        x = x * rnge + min_max[0]

    return x


def closest_tokens(emb, word_embeddings, tokenizer, n=1):      
# This finds the n tokens in the vocabulary that are closest in the embedding space (in terms of Euclidean distance) to a given word embedding (‘emb’).
# Note that here 'emb' may or may not correspond to a token (i.e., it may or may not be a 'legal' embedding).
# Function returns a 4-tuple (list of the n tokens, list of their indices, list of their distances from emb, and list of their embedding vectors)
    torch.cuda.empty_cache()
    dists = torch.linalg.norm(word_embeddings - emb, dim=1)
    sorted_dists, ix = torch.sort(dists)	 
    # sorted_dists is a list of all embedding distances from 'emb', across entire vocab, sorted in increasing order, 
    # ix is a list of their corresponding 'vocab indices'
    tokens = [tokenizer.decode(i) for i in ix[:n]]
    # For each of the first n 'vocab indices' in ix, we decode it into the string version of the corresponding token. 
    # These strings then constitute the list 'tokens'.
    ixs = ix[:n]
    dists = sorted_dists[:n]
    embs = word_embeddings[ixs]  # Each of these n 'embeddings' is a tensor of shape (768,)
    return tokens, ixs, dists, embs  


def model_emb(model, inputs_embeds, word_embeddings, output_len):
# 'input_embeds' is a tensor of shape (batch_size, input_len, embedding_dim)
# 'output_len' is an integer specifying the number of output tokens to generate
# Note that this function doesn't involve a target output. It simply takes a tensor of input embeddings (based on input length),
# calculates perplexities for that batch of input sequences,
# and runs the batch of input sequences through GPT2, for each finding next tokens iteratively 'output_len' number of times
    embs = inputs_embeds   # This is going to get expanded using 'output_embs'
    logits = []
    ixs = []
    input_logits = None
    for i in range(output_len):
        model_out = model(inputs_embeds=embs, return_dict=True)
        # Does a forward pass of GPT2 (or whichever model) on a batch of inputs (given as a tensor 'embs' of embeddings).
        # This 'embs' will expand along its 1st dimension with each iteration.
        # Outputs logits and more (hidden states, attention, etc.) as a dictionary 'model_out'.
        # But we'll only be concerned with model_out.logits.

        if i == 0:
            input_logits = model_out.logits 
            # On first pass through loop, we simply use the logits of the model output
            # That's a tensor of shape (batch_size, input_len, vocab_size) giving logits for each input in each batch.
            # Presumably for each input, this is conditioned on the inputs that preceded it?

        # On every pass throught the loop (including the first), we defined this tensor of shape (batch_size, 1, vocab_size):
        last_logits = model_out.logits[:,-1].unsqueeze(1)  
        # model_out.logits[:,-1] will be a 2D tensor of shape (batch_size, vocab_size), just giving logits for last input/embedding across all batches/tokens
        # unsqueezing, we get tensor of shape (batch_size, 1, vocab_size) also giving logits of last input/embedding, differently formatted  
        logits.append(last_logits)  # appends last_logits tensor to the 'logits' list 
        ix = torch.argmax(last_logits, dim=-1)  # for each batch, finds the vocab index of the token with the largest logit in last_logits
        ixs.append(ix) # ...and appends this tensor of shape (batch_size,) (containing indices) it to the list 'ixs'
        output_embs = word_embeddings[ix]   # for each batch, finds embedding for the token with that index...
        embs = torch.cat([embs, output_embs], dim=1)  #...concatenates that tensor of embeddings to the 'embs' tensor in the first dimension before next iteration

     # When the loop is completed 'embs' will be a tensor containing all of the input and output word embeddings produced by the model   
     # ...so presumably of shape (batch_size, input_len + output_len, embedding_dim)

    logits = torch.cat(logits, dim=1)   # this converts logits from a list of tensors to a single tensor, by concatenating all of the tensors in the list
                                        # it will have shape (batch_size, output_len, vocab_size)
    perp = perplexity(torch.cat([input_logits, logits], dim=1))    
    return logits, embs, perp          
    # logits has shape (batch_size, output_len, vocab_size),         CHECK THAT!
    # embs has shape (batch_size, input_len + output_len, embedding_dim)
    # perp has shape (batch_size,)


def perplexity(logits):
    # logits is of shape (batch_size, 'sequence length', vocab_size)
    # for all current calls, 'sequence length' is going to be input_len
    probs, ix = torch.max(torch.softmax(logits, dim=-1), dim=-1)
    # torch.softmax(logits, dim=-1) will also be a tensor of shape (batch_size, 'sequence length', vocab_size), 
    # but where the logits in the last dimension get converted into probabilities via softmax. torch.max() then pull out the largest of these and its index
    # probs is a tensor that contains the maximum probability for each token in the embedding sequence, shape (batch_size, 'sequence length')
    # ix is a tensor that contains the corresponding indices, also with shape (batch_size, 'sequence length')
    perp = 1/ (torch.prod(probs, dim=-1)**(1/probs.shape[-1])) - 1
    # defines a scalar that's larger with greater uncertainty (so if the probs are small, their product is small, the reciprocal of some power is large)
    # probs.shape[-1] is output_len; the idea of raising the probs product to power 1/output_len is to make perplexities comparable across different output lengths
    return perp

