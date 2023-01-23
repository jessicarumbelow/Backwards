from helpers import *

import torch
from matplotlib import pyplot as plt
from IPython import display
import numpy as np
import argparse
import json
import os
from collections import Counter
import random

os.environ["WANDB_API_KEY"] = "4c2a9ff74fdb68f1f92a87d2ff834315f06a3530"
os.environ["WANDB_SILENT"] = "true"


# Here's the key function that optimises for a sequence of input embeddings, given a target_output string:
def optimise_input(model,
                   word_embeddings,
                   tokenizer,
                   device,
                   epochs=100, 
                   lr=0.1, 
                   no_reinit=False,    # Do we re-initialise inputs tensor with random entries when an optimal input is found?
                   w_freq=10,           # logging (write) frequency
                   rand_input=False,      # If False, start_inputs will be entirely random, if true cluster centroids get used.
                   batch_size=20, 
                   input_len=10, 
                   target_output=' world',    # Default target output is the "." token; this won't generally be used
                   output_len=None,
                   dist_reg=0.1,       # distance regularisation coefficient
                   perp_reg=0,       # perplexity regularisation coefficient; setting to 0 means perplexity loss isn't a thing
                   loss_type='log_prob_loss', 
                   seed=0,
                   return_early=False,    # finishes if single optimised input is found
                   verbose=1,            # Controls how much info gets printed.
                   lr_decay=False,       # Use learning rate decay? If so, a scheduler gets invoked.
                   run_random=0,
                   distance_type='cosine',
                   equal_clusters=False,
                   penalise_repetition=False,
                   **kwargs): 

    # Picks a single token at random from vocabulary for target_output
    if run_random > 0:
        random_ix = (torch.rand(1)*word_embeddings.shape[0]).int()
        target_output = tokenizer.decode(random_ix)  # Converts token index to string representation
        wandb.config.update({'target_output':target_output}, allow_val_change=True)
    
    print('Optimising input of length {} to maximise output logits for "{}"'.format(input_len, target_output))
    done = None

    output_ix = tokenizer.encode(target_output, return_tensors='pt')[0].to(device)
    # output_ix is a 1-D tensor of shape (output_len,) that contains the indices of the tokens in the encoding of the string 'target_output'
    # tokenizer.encode(target_output, return_tensors='pt') is a list containing this one tensor, hence the need for the [0]
    # "return_tensors='pt'" ensures that we get a tensor in PyTorch format

    optimised_inputs = set()
    optimised_tokens = []
    metrics_table = wandb.Table(columns=['Input', 'Output', 'Loss','Perplexity', 'Distance', 'Probs'])

    if output_len == None or output_len < output_ix.shape[0]:   # If we don't specify output_len (i.e. it's == None), then...
        output_len = output_ix.shape[0]       # ...it will be set to the number of tokens in the encoding of the string 'target_output'
    else:
        possible_target_positions = torch.stack([torch.arange(0, output_ix.shape[0]) + i for i in range(output_len - output_ix.shape[0] + 1)])
        # generates list of legal token positions within output ("sequentiality enforcer")

    if rand_input == True:
        start_input = torch.rand(batch_size, input_len, word_embeddings.shape[-1]).to(device)
        # If no base_input is provided, we construct start_input as a random tensor 
        # of shape (batch_size, input_len, embedding_dim)
        start_input = normalise(start_input,[word_embeddings.min(dim=0)[0], word_embeddings.max(dim=0)[0]])
        # We normalise this random tensor so that its minimum and maximum values correspond columnwise to those in the entire word_embeddings tensor
        # This keeps each dimension of the starting embeddings within the range of the legal token embeddings. 
    else:
        # Otherwise we use k-means clustering to find centroids as our start_input embeddings
        num_clusters = batch_size*input_len
        _, centroids = kkmeans(word_embeddings.detach(), num_clusters, seed=seed, distance_type=distance_type, equal_clusters=equal_clusters)
        start_input = centroids.reshape(batch_size, input_len, -1)

    input = torch.nn.Parameter(start_input, requires_grad=True)
    # input is Parameter object that wraps a tensor and adds additional functionality. 
    
    optimiser = torch.optim.Adam([input], lr=lr)
    # standard optimiser; note that it generally operates on a list of tensors, so we're giving it a list of one tensor; standard learning rate
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, 'min', patience=20, cooldown=20, factor=0.5)
    # this is used when loss hasn't improved for 20 timesteps; this scheduler will reduce the lr by a 'factor' of 0.5 when the 
    # validation loss stops improving for 'patience' (here 20) epochs, and will wait 'cooldown' (here 20) epochs before resuming normal operation.

    for e in range(epochs):
        logits, emb, perp = model_emb(model, torch.clamp(input, word_embeddings.min(dim = 0)[0], word_embeddings.max(dim = 0)[0]), word_embeddings, output_len)
        # Does forward pass on a 'clamped' version of the 'input' tensor (which constrains it for each dimension to the range of all token embeddings)
        # Iterates to produce an output of output_len tokens, 
        # returns: 'logits' = tensor of logits for output, of shape (batch_size, output_len, vocab_size)
        # 'emb': tensor of embeddings for input+output of shape (batch_size, input_len + output_len, embedding_dim); 
        # 'perp': the input sequence perplexities tensor, of shape (batch_size,)


        probs = torch.softmax(logits, dim=-1)
        # For each batch, output, converts the sequence of logits (of length 'vocab_size') in the 'logits' tensor to probabilities, using softmax

        perp_loss = perp.mean()  # across all elements in the batch

        if output_len > output_ix.shape[0]:
            target_logits = logits[:,possible_target_positions,output_ix].max(dim=1)[0]
            target_probs = probs[:,possible_target_positions,output_ix].max(dim=1)[0]
            # for 'contains in' scenario
            # find the position (among all legal positions) for the target output that gives highest logits
            # and analogously with probs

        else:
            target_logits = logits[:,torch.arange(output_len), output_ix]
            target_probs = probs[:,torch.arange(output_len), output_ix]

            # This handles case where output_len == output_ix.shape[0]
            # target_logits now of shape (batch_size, output_len)

        # consolidation so that closest_tokens only needs to be used in a single loop
        token_dist, closest_ix = [],[]
        for b in input:
            tds, cixs = [], []
            for be in b:
                _, cix, td, _ = closest_tokens(be, word_embeddings, tokenizer, distance_type=distance_type)
                tds.append(td)
                cixs.append(cix)
            token_dist.append(torch.stack(tds))
            closest_ix.append(torch.stack(cixs))

        # convert lists into tensors
        token_dist, closest_ix = torch.stack(token_dist).squeeze(-1), torch.stack(closest_ix).squeeze(-1)

        # This creates a tensor of shape (batch_size, input_len, 1) which gives mean distance to nearest
        # legal token embedding across all input embeddings in each batch
        mean_token_dist = token_dist.mean() 

        # log_prob_loss is the current default.
        if loss_type == 'log_prob_loss':
            loss = -torch.log(target_probs)
        elif loss_type == 'CE':
            loss = torch.nn.functional.cross_entropy(logits.swapaxes(-1,-2), output_ix.repeat(batch_size, 1), reduction='none')
        else:
            print(loss_type + 'is not implemented.')
            return 

        batch_loss = loss.mean()

        total_loss = torch.stack([mean_token_dist * dist_reg, batch_loss, perp_loss * perp_reg]).mean()
        
        if penalise_repetition:
            rep_penalty = logits[:,:input_len, output_ix].sum()
            total_loss += rep_penalty
        else:
            rep_penalty = 0


        model_outs = model.generate(closest_ix, max_length = output_len+input_len)
        # The 'closest_ix' tensor is passed as the initial input sequence to the model, 
        # and the max_length parameter specifies the maximum length of the total sequence to generate.
        # The output sequence will be terminated when the maximum length is reached.
        # 
        # The output of the model.generate method will be a tuple containing the generated sequences and the model's internal states. 
        # The generated sequences will be stored in a tensor of shape (batch_size, output_len+input_len). 
        # Each element of the tensor will be a sequence of tokens with a length of at most output_len+input_len.
        
        for b in range(batch_size):
            if target_output in tokenizer.decode(model_outs[b][input_len:]) and tokenizer.decode(model_outs[b]) not in optimised_inputs:
                optimised_tokens += [tokenizer.decode(t) for t in model_outs[b][:input_len]]

                counts = Counter(optimised_tokens)
                labels, values = zip(*counts.items())

                data = [[label, val] for (label, val) in zip(labels, values)]
                table = wandb.Table(data=data, columns = ["Token", "Count"])
                wandb.log({"token_freqs" : wandb.plot.bar(table, "Token",
                               "Count", title="Token Freqs")})


                done = tokenizer.decode(model_outs[b])
                optimised_inputs.add(done)
                metrics_table.add_data(*[tokenizer.decode(model_outs[b][:input_len]), tokenizer.decode(model_outs[b][input_len:])] + torch.stack([loss.squeeze(-1)[b], perp[b], token_dist.mean(dim=1)[b]], dim=-1).tolist() + target_probs[b].tolist())
                wandb.log({'Optimised Inputs': wandb.Html(''.join(['<p>{}.{}</p>'.format(i, repr(s)) for i, s in enumerate(optimised_inputs)]))})

                if no_reinit == False:
                    if rand_input == True:
                        input.data[b] = normalise(torch.rand_like(input[b]),[word_embeddings.min(dim=0)[0], word_embeddings.max(dim=0)[0]])
                    else:
                        rand_centroids = centroids[np.random.randint(0, batch_size, size=input_len)].unsqueeze(0)
                        input.data[b] = rand_centroids
                    # Random re-initialisation (if 'rand_after' set to True)
        
        if ((e+1) % w_freq == 0) or done and return_early:
        # Every w epochs we print, unless we have found an optimised input before that and 'return_early' == True. 
        # We use return_early == True if we want to find just one optimised input.
             
            print("Optimised Inputs:", optimised_inputs)
            print('{}/{} Output Loss: {} Emb Dist Loss: {} Perp Loss: {} LR: {}'.format(e+1, epochs, batch_loss, mean_token_dist, perp_loss, optimiser.param_groups[0]['lr']))
            if verbose == 3:
                print('Target Probs: {}\nTarget Logits: {}\nInput Dists: {}\nInput Perplexity: {}\n'.format(target_probs.detach().cpu().numpy(), target_logits.detach().cpu().numpy(), token_dist.detach().cpu().numpy(), perp.detach().reshape(-1).cpu().numpy()))
            # Optimised inputs and additional information are printed as part of log

            closest_embeddings = []

            for b in range(batch_size):
                if verbose > 0:
                    if verbose == 2:
                        print(b, repr(' Raw embeddings: {}'.format(''.join([closest_tokens(e)[0][0] for e in emb[b]]))))

                    print(b, repr(' Closest embeddings: {}'.format(tokenizer.decode(model_outs[b]), '\n')))
                    closest_embeddings.append(tokenizer.decode(model_outs[b]))

            wandb.log({'Closest Embeddings': wandb.Html(''.join(['<p>{}.{}</p>'.format(i, repr(ce)) for i, ce in enumerate(closest_embeddings)])), 'Total Loss':total_loss, 'Mean Token Distance': mean_token_dist, 'Mean Loss': batch_loss, 'Mean Perplexity Loss':perp_loss, 'Epoch':e, 'LR':optimiser.param_groups[0]['lr'], 'Num Inputs Found':len(optimised_inputs), 'Repetition Penalty':rep_penalty})

            if done and return_early:
                print('\nOptimised Input: "{}"'.format(done))
                return {'Metrics':metrics_table}
                # we know optimised_inputs set contains a single element in this case
            
        optimiser.zero_grad()
        total_loss.backward()
        optimiser.step()
        # standard NN optimisation

        if lr_decay:
            scheduler.step(total_loss)
        done = None

    return {'Metrics':metrics_table}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb_user', type=str, default='jessicamarycooper')
    parser.add_argument('--model_name', type=str, default='gpt2')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--no_reinit', action='store_true')
    parser.add_argument('--w_freq', type=int, default=10)
    parser.add_argument('--rand_input', action='store_true')
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--input_len', type=int, default=10)
    parser.add_argument('--target_output', type=str, default=' world')
    parser.add_argument('--output_len', type=int)
    parser.add_argument('--dist_reg', type=float, default=0.1)
    parser.add_argument('--perp_reg', type=float, default=0)
    parser.add_argument('--loss_type', type=str, default='log_prob_loss')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--return_early', action='store_true')
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--lr_decay', action='store_true')
    parser.add_argument('--note', type=str, default='')
    parser.add_argument('--run_test_set', type=int, default=-1)
    parser.add_argument('--run_random', type=int, default=0)
    parser.add_argument('--distance_type', type=str, default='cosine')
    parser.add_argument('--equal_clusters', action='store_true')
    parser.add_argument('--penalise_repetition', action='store_true')


    
    args = parser.parse_args()

    test_sets = [[' externalToEVA', 'quickShip', ' TheNitrome', 'embedreportprint', 'rawdownload', 'reportprint', ' サーティ', ' RandomRedditor', 'oreAndOnline', 'InstoreAndOnline', ' externalTo', 'StreamerBot', 'ActionCode', 'Nitrome'],
                [' girl', ' boy', ' woman', ' man', ' good', ' evil', ' white', ' black', ' doctor', ' England', ' USA'],
                ]
    torch.manual_seed(args.seed)
    random.seed(0)
    np.random.seed(0)

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('Using {} device.'.format(args.device))

    args.model, args.word_embeddings, args.tokenizer = load_all(args.model_name, args.device)

    if args.run_test_set > -1:
        for to in test_sets[args.run_test_set]:
            args.target_output = to
            run = wandb.init(config=args, project='backwards', entity=args.wandb_user, reinit=True)
            results = optimise_input(**vars(args))
            wandb.log(results)
            run.finish()

    if args.run_random > 0:

        seeds = (torch.rand(args.run_random)*60000).int()
        for r in range(args.run_random):
            args.seed=seeds[r]
            args.target_output = 'RANDOM'
            run = wandb.init(config=args, project='backwards', entity=args.wandb_user, reinit=True)
            results = optimise_input(**vars(args))
            wandb.log(results)
            run.finish()

    if args.run_test_set == -1 and args.run_random == 0:
        run = wandb.init(config=args, project='backwards', entity=args.wandb_user, reinit=True)
        results = optimise_input(**vars(args))
        wandb.log(results)
        run.finish()

