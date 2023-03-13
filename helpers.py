import subprocess
import sys
import os
import torch


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--upgrade"])

try:
    from transformers import GPT2Tokenizer, GPT2LMHeadModel, utils, AutoTokenizer, GPTJForCausalLM
except:
    install('transformers')
    install('accelerate')
    from transformers import GPT2Tokenizer, GPT2LMHeadModel, utils, AutoTokenizer, GPTJForCausalLM

try:
    import wandb
except:
    install('wandb')
    import wandb

utils.logging.set_verbosity_error()


def load_all(model_name="gpt2", device='cpu', save_dir=''):
    print(save_dir)
    if save_dir == '':
        cur_dir = os.listdir()
    else:
        cur_dir = os.listdir(save_dir)

    if model_name + '_tokenizer' in cur_dir:
        print('Loading tokenizer...')
        tokenizer = torch.load(save_dir + model_name + '_tokenizer')
    else:
        print('Downloading tokenizer...')
        if 'gpt-j' in model_name:
            print('WARNING: I haven\'t tuned hyperparameters for gpt-j. Don\'t expect it to work very well on defaults!')
            tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
        else:
            tokenizer = GPT2Tokenizer.from_pretrained(model_name, padding_side='left')
        torch.save(tokenizer, save_dir + model_name + '_tokenizer')
    pad_token_id = tokenizer.eos_token_id

    if model_name + '_model' in cur_dir:
        print('Loading model...')
        model = torch.load(save_dir + model_name + '_model').to(device)
    else:
        print('Downloading model...')

        if 'gpt-j' in model_name:
            model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", revision="float16",
                                                    torch_dtype=torch.float16, low_cpu_mem_usage=True)
        else:
            model = GPT2LMHeadModel.from_pretrained(model_name, pad_token_id=tokenizer.eos_token_id)
        torch.save(model, save_dir + model_name + '_model')
    model.eval()


    embeddings = model.transformer.wte.weight.detach()
    if model_name + '_embeddings' not in cur_dir:
        torch.save(embeddings, save_dir + model_name + '_embeddings')

    return model.to(device), embeddings.to(device), tokenizer


def kkmeans(embeddings, num_clusters, threshold=0.00001, max_iter=1000, seed=0, overwrite=False,
            save_dir='', equal_clusters=False, cluster_dim=-1):

    if cluster_dim != -1 and equal_clusters:
        print('WARNING! Equal clusters not supported for dimension clustering.')
    embeddings = embeddings.detach()

    centroid_fname = str(embeddings.shape[0]) + '_' + str(embeddings.shape[1]) + '_' + str(num_clusters) + '_' + str(
        equal_clusters) + '_' + str(seed) + ' dim' + str(cluster_dim) + '_centroids'
    cluster_fname = str(embeddings.shape[0]) + '_' + str(embeddings.shape[1]) + '_' + str(num_clusters) + '_' + str(
        equal_clusters) + '_' + str(seed) + ' dim' + str(cluster_dim) + '_cluster'

    if not overwrite:
        cur_dir = os.listdir()
        if centroid_fname in cur_dir:
            print('Loading clusters...')
            return torch.load(cluster_fname), torch.load(centroid_fname)

    print('Finding clusters...')
    if seed != -1:
        torch.manual_seed(seed)
    cluster_size = embeddings.shape[0] // num_clusters
    # initial centroids is a set of random token embeddings (one for each cluster)
    centroids = embeddings[torch.randperm(embeddings.shape[0])[:num_clusters]]

    movement = 9999  # this will be used in each iteration step as mean centroid movement distance
    i = 0

    while movement > threshold and i < max_iter:
        i += 1

        print(embeddings.shape, centroids.shape)
        if cluster_dim > -1:
            distances = 1 - (embeddings[:, cluster_dim] @ centroids[cluster_dim].T)

        else:
            distances = 1 - (embeddings @ centroids.T)

        closest_distance, closest_centroid = torch.sort(distances, dim=-1)
        clusters = [embeddings[(closest_centroid[:, 0] == i)] for i in range(num_clusters)]

        if equal_clusters:
            for c in range(num_clusters):
                if clusters[c].shape[0] > cluster_size:
                    # sort cluster embs by distance from centroid so spares are furthest away
                    _, sorted_cluster_embs_ix = torch.sort(
                        1 - (clusters[c] @ clusters[c].mean(dim=0).unsqueeze(0).T).squeeze(-1))

                    clusters[c] = clusters[c][sorted_cluster_embs_ix]
                    spare_embs = clusters[c][cluster_size:]
                    clusters[c] = clusters[c][:cluster_size]
                    for cc in range(num_clusters):
                        if clusters[cc].shape[0] < cluster_size:

                            _, sorted_spare_embs_ix = torch.sort(
                                1 - (spare_embs @ clusters[cc].mean(dim=0).unsqueeze(0).T).squeeze(-1))

                            free_space = cluster_size - clusters[cc].shape[0]
                            clusters[cc] = torch.cat([clusters[cc], spare_embs[sorted_spare_embs_ix][:free_space]])
                            spare_embs = spare_embs[free_space:]

        new_centroids = torch.stack([c.mean(dim=0)/torch.sqrt(torch.sum(c.mean(dim=0)**2, dim=-1, keepdim=True)) for c in clusters])
        movement = torch.abs(new_centroids - centroids).mean()
        print('Movement :', movement)
        centroids = new_centroids

    centroids = torch.stack([c.mean(dim=0)/torch.sqrt(torch.sum(c.mean(dim=0)**2, dim=-1, keepdim=True)) for c in clusters])
    print([c.shape[0] for c in clusters])
    torch.save(clusters, save_dir + cluster_fname)
    torch.save(centroids, save_dir + centroid_fname)
    return clusters, centroids


def normalise(x, min_max=[]):

    rnge = x.max() - x.min()
    if rnge > 0:
        x = (x - x.min()) / rnge

    if len(min_max) > 1:
        rnge = min_max[1] - min_max[0]
        x = x * rnge + min_max[0]

    return x


def closest_tokens(emb, word_embeddings, tokenizer, n=1):
    torch.cuda.empty_cache()
    dists = 1 - (emb.unsqueeze(0) @ word_embeddings.T).squeeze(0)
    sorted_dists, ix = torch.sort(dists)

    tokens = [tokenizer.decode(i) for i in ix[:n]]
    ixs = ix[:n]
    dists = sorted_dists[:n]
    embs = word_embeddings[ixs]  
    return tokens, ixs, dists, embs


def model_emb(model, inputs_embeds, word_embeddings, output_len):

    embs = inputs_embeds 
    logits = []
    ixs = []
    input_logits = None
    for i in range(output_len):
        model_out = model(inputs_embeds=embs, return_dict=True)

        if i == 0:
            input_logits = model_out.logits[:, :-1]

        last_logits = model_out.logits[:, -1].unsqueeze(1)
        logits.append(last_logits) 
        ix = torch.argmax(last_logits,
                          dim=-1)  
        ixs.append(ix) 
        output_embs = word_embeddings[ix] 
        embs = torch.cat([embs, output_embs],
                         dim=1) 

    logits = torch.cat(logits,
                       dim=1)  
    perp = perplexity(torch.cat([input_logits, logits], dim=1))
    return logits, embs, perp
    


def perplexity(logits):
    probs, ix = torch.max(torch.softmax(logits, dim=-1), dim=-1)
   
    perp = 1 / (torch.prod(probs, dim=-1) ** (1 / probs.shape[-1])) - 1
    return perp

