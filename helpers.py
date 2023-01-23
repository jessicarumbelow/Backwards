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

try:
    import wandb
except:
    install('wandb')
    import wandb

utils.logging.set_verbosity_error()

def load_all(model_name="gpt2", device='cpu'):
    cur_dir = os.listdir()
    
    if model_name + '_tokenizer' in cur_dir:
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


    if model_name + '_model' in cur_dir:
        print('Loading model...')
        model = torch.load(model_name + '_model').to(device)
    else:
        print('Downloading model...')

        if 'gpt-j' in model_name:
            model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B").to(device)
        else:
            model = GPT2LMHeadModel.from_pretrained(model_name, pad_token_id=tokenizer.eos_token_id).to(device)
        torch.save(model, model_name + '_model')
    model.eval()

    # 'embeddings' tensor gives emeddings for each token in the vocab for this model,
    # has shape (vocab length, embedding dimension) 
    
    embeddings = model.transformer.wte.weight.to(device)
    if model_name + '_embeddings' not in str(cur_dir):
        torch.save(embeddings, model_name + '_embeddings')

    return model, embeddings, tokenizer

def kkmeans(embeddings, num_clusters, threshold=0, max_iter=300, seed=-1, distance_type='cosine', overwrite=False, save_dir='', equal_clusters=False):
     
    def dist(embeddings, centroids):
        if distance_type == 'cosine':
            distances = 1-cos_sim(embeddings, centroids)
        else:
            distances = torch.cdist(embeddings, centroids, p=2)
        return distances

        
    centroid_fname = str(embeddings.shape) + '_' + str(num_clusters) + '_' + str(seed) + '_e' + str(equal_clusters) + '_' + distance_type + '_centroids'
    cluster_fname = str(embeddings.shape) + '_' + str(num_clusters) + '_' + str(seed) + '_e' + str(equal_clusters) + '_' + distance_type + '_cluster'

    if not overwrite:
        cur_dir = os.listdir()
        if centroid_fname in cur_dir:
            print('Loading clusters...')
            return torch.load(cluster_fname), torch.load(centroid_fname)

    print('Finding clusters...')
    if seed != -1:
        torch.manual_seed(seed) 
    cluster_size = embeddings.shape[0]//num_clusters
    # initial centroids is a set of random token embeddings (one for each cluster)
    centroids = embeddings[torch.randperm(embeddings.shape[0])[:num_clusters]]

    movement = 9999  #this will be used in each iteration step as mean centroid movement distance
    i = 0

    while movement > threshold and i < max_iter: 
        i += 1

        # (vocab_len, num_clusters) Euclidean distances of all token embeddings from each of the centroids.
        distances = dist(embeddings, centroids)
        
        #(vocab_len, num_cluster), for each token embedding recording the sorted distances to each centroid, and the corresponding sorted centroid indexes.
        closest_distance, closest_centroid = torch.sort(distances, dim=-1)
        clusters = [embeddings[(closest_centroid[:,0]==i)] for i in range(num_clusters)]

        if equal_clusters:
            for c in range(num_clusters):
                if clusters[c].shape[0] > cluster_size:
                    #sort cluster embs by distance from centroid so spares are furthest away
                    _, sorted_cluster_embs_ix = torch.sort(dist(clusters[c], clusters[c].mean(dim=0).unsqueeze(0)).squeeze(-1))
                    clusters[c] = clusters[c][sorted_cluster_embs_ix]
                    spare_embs = clusters[c][cluster_size:]
                    clusters[c] = clusters[c][:cluster_size]
                    for cc in range(num_clusters):
                        if clusters[cc].shape[0] < cluster_size:
                            #sort spare embs by distance from current cluster centroid so nearest ones are added 
                            _, sorted_spare_embs_ix = torch.sort(dist(spare_embs, clusters[cc].mean(dim=0).unsqueeze(0)).squeeze(-1))
                            free_space = cluster_size - clusters[cc].shape[0]
                            clusters[cc] = torch.cat([clusters[cc], spare_embs[sorted_spare_embs_ix][:free_space]])
                            spare_embs = spare_embs[free_space:]
        
        
        new_centroids = torch.stack([c.mean(dim=0) for c in clusters])
        movement = torch.abs(new_centroids - centroids).mean()
        print(movement)
        centroids = new_centroids

    centroids = torch.stack([c.mean(dim=0) for c in clusters])
    print([c.shape[0] for c in clusters])
    torch.save(clusters, save_dir + cluster_fname)
    torch.save(centroids, save_dir + centroid_fname)
    return clusters, centroids



def normalise(x, min_max=[]):     
# normalises values of (array or tensor) x according to first (min) and second (max) values in list min_max. 
# This effectively defaults to [0,1] if the list doesn't contain exactly two elements. 

# First normalise x to [0,1]
    rnge = x.max() - x.min()
    if rnge > 0:
        x = (x - x.min())/rnge

# Now, if there's a min and max given in min_max list, multiply by difference and add minimum
    if len(min_max) > 1:
        rnge = min_max[1] - min_max[0]
        x = x * rnge + min_max[0]

    return x


def cos_sim(A, B, dim=1, eps=1e-8):
    #https://stackoverflow.com/a/72369507
      numerator = A @ B.T
      A_l2 = torch.mul(A, A).sum(axis=dim)
      B_l2 = torch.mul(B, B).sum(axis=dim)
      denominator = torch.max(torch.sqrt(torch.outer(A_l2, B_l2)), torch.tensor(eps))
      return torch.div(numerator, denominator)


def closest_tokens(emb, word_embeddings, tokenizer, n=1, distance_type='cosine'):      
# This finds the n tokens in the vocabulary that are closest in the embedding space (in terms of Euclidean distance) to a given word embedding (‘emb’).
# Note that here 'emb' may or may not correspond to a token (i.e., it may or may not be a 'legal' embedding).
# Function returns a 4-tuple (list of the n tokens, list of their indices, list of their distances from emb, and list of their embedding vectors)
    torch.cuda.empty_cache()
    
    if distance_type=='cosine':
        dists = 1-cos_sim(emb.unsqueeze(0), word_embeddings).squeeze(0).squeeze(0)
    else:
        dists = torch.linalg.norm(word_embeddings - emb, dim=1)

    sorted_dists, ix = torch.sort(dists)

    # sorted_dists is a list of all embedding distances from 'emb', across entire vocab, sorted in increasing order, 
    # ix is a list of their corresponding 'vocab indices'
    tokens = [tokenizer.decode(i) for i in ix[:n]]
    # For each of the first n 'vocab indices' in ix, we decode it into the string version of the corresponding token. 
    # These strings then constitute the list 'tokens'.
    ixs = ix[:n]
    dists = sorted_dists[:n]
    embs = word_embeddings[ixs]  # Each of these n 'embeddings' is a tensor of shape (emedding dimension,)
    return tokens, ixs, dists, embs  


def model_emb(model, inputs_embeds, word_embeddings, output_len):
# 'input_embeds' is a tensor of shape (batch_size, input_len, embedding_dim)
# 'output_len' is an integer specifying the number of output tokens to generate
# Note that this function doesn't involve a target output. It simply takes a tensor of input embeddings (based on input length),
# and runs the batch of input sequences through the model, for each, finding next tokens iteratively 'output_len' number of times
# and calculates perplexities for that batch of input sequences + output tokens
    embs = inputs_embeds   # This tensor is going to get expanded using 'output_embs'
    logits = []
    ixs = []
    input_logits = None
    for i in range(output_len):
        model_out = model(inputs_embeds=embs, return_dict=True)
        # Does a forward pass of the model on a batch of inputs (given as a tensor 'embs' of embeddings).
        # This 'embs' will expand along its 1st dimension with each iteration.

        if i == 0:
            input_logits = model_out.logits[:,:-1]
            # That's a tensor of shape (batch_size, input_len, vocab_size) giving logits for each input in each batch.
            # For all further passes of the loops, only last token logits are relevant.  

        # On every pass throught the loop (including the first), we define this tensor of shape (batch_size, 1, vocab_size):
        last_logits = model_out.logits[:,-1].unsqueeze(1)  
        # model_out.logits[:,-1] will be a 2D tensor of shape (batch_size, vocab_size), just giving logits for last embedding across all batches/tokens
        # unsqueezing, we get tensor of shape (batch_size, 1, vocab_size) also giving logits of last input/embedding, differently formatted  
        logits.append(last_logits)  # appends last_logits tensor to the 'logits' list 
        ix = torch.argmax(last_logits, dim=-1)  # for each batch, finds the vocab index of the token with the largest logit in last_logits
        ixs.append(ix) # ...and appends this tensor of shape (batch_size,) (containing indices) it to the list 'ixs'
        output_embs = word_embeddings[ix]   # for each batch, finds embedding for the token with that index...
        embs = torch.cat([embs, output_embs], dim=1)  #...concatenates that tensor of embeddings to the 'embs' tensor in the first dimension before next iteration

     # When the loop is completed 'embs' will be a tensor containing all of the input and output word embeddings produced by the model   
     # ...so of shape (batch_size, input_len + output_len, embedding_dim)

    logits = torch.cat(logits, dim=1)   # this converts logits from a list of tensors to a single tensor, by concatenating all of the tensors in the list

    perp = perplexity(torch.cat([input_logits, logits], dim=1))    
    return logits, embs, perp          
    # logits has shape (batch_size, output_len, vocab_size),        
    # embs has shape (batch_size, input_len + output_len, embedding_dim)
    # perp has shape (batch_size,)


def perplexity(logits):

    probs, ix = torch.max(torch.softmax(logits, dim=-1), dim=-1)
    # torch.softmax(logits, dim=-1) will also be a tensor of shape (batch_size, 'sequence length', vocab_size), 
    # but where the logits in the last dimension get converted into probabilities using softmax then pull out the largest of these and its index
    # probs is a tensor that contains the maximum probability for each token in the embedding sequence, shape (batch_size, 'sequence length')
    # ix is a tensor that contains the corresponding indices, also with shape (batch_size, 'sequence length')
    perp = 1/ (torch.prod(probs, dim=-1)**(1/probs.shape[-1])) - 1
    # defines a scalar that's larger with greater uncertainty (so if the probs are small, their product is small, the reciprocal of some power is large)
    # probs.shape[-1] is sequence length; the idea of raising the probs product to power 1/sequence length is to make perplexities comparable across different output lengths
    # subtracting 1 guarantees perplexity 0 in limit of case of total certainty

    return perp

