from helpers import *
!pip install wandb
import torch
from matplotlib import pyplot as plt
from IPython import display
import numpy as np
import argparse
import json
import os

os.environ["WANDB_API_KEY"] = "4c2a9ff74fdb68f1f92a87d2ff834315f06a3530"
os.environ["WANDB_SILENT"] = "true"


# Here's the key function that optimises for a sequence of input embeddings, given a target_output string:
def optimise_input(model,
                   word_embeddings,
                   tokenizer,
                   device,
                   epochs=100, 
                   lr=0.1, 
                   rand_after=False,    # Do we re-initialise inputs tensor with random entries when an optimal input is found?
                   w_freq=10,           # logging (write) frequency
                   base_input=False,      # If False, start_inputs will be entirely random
                   batch_size=20, 
                   input_len=3, 
                   target_output='.',    # Default target output is the "." token; this won't generally be used
                   output_len=None,
                   dist_reg=1,       # distance regularisation coefficient
                   perp_reg=0,       # perplexity regularisation coefficient; setting to 0 means perplexity loss isn't a thing
                   loss_type='log_prob_loss', 
                   seed=0,
                   return_early=True,    # finishes if single optimised input is found
                   verbose=1,            # Controls how much info gets logged.
                   lr_decay=False,       # Use learning rate decay? If so, a scheduler gets invoked.
                   run_random=0,
                   **kwargs): 

    torch.manual_seed(args.seed)

    if run_random > 0:
        random_ix = (torch.rand(1)*word_embeddings.shape[0]).int()
        target_output = tokenizer.decode(random_ix)
        wandb.config.update({'target_output':target_output}, allow_val_change=True)
    
    print('Optimising input of length {} to maximise output logits for "{}"'.format(input_len, target_output))

    done = None

    output_ix = tokenizer.encode(target_output, return_tensors='pt')[0].to(device)
    # output_ix is a 1-D tensor of shape (output_len,) that contains the indices of the tokens in the encoding of the string 'target_output'
    # tokenizer.encode(target_output, return_tensors='pt') is a list containing this one tensor, hence the need for the [0]
    # "return_tensors='pt'" ensures that we get a tensor in PyTorch format

    optimised_inputs = set()
    metrics_table = wandb.Table(columns=['Input', 'Output', 'Loss','Perplexity', 'Distance'])

    if output_len == None or output_len < output_ix.shape[0]:                    # This won't generally be the case, but if we don't specify output_len (i.e. it's == None), then...
        output_len = output_ix.shape[0]       # ...it will be set to the number of tokens in the encoding of the string 'target_output'
    else:
        possible_target_positions = torch.stack([torch.arange(0, output_ix.shape[0]) + i for i in range(output_len - output_ix.shape[0] + 1)])

    # Typically this would print something like 'Optimising input of length 6 to maximise output logits for "KILL ALL HUMANS!"'.

    if base_input == False:
        start_input = torch.rand(batch_size, input_len, word_embeddings.shape[-1]).to(device)
        # If no base_input is provided, we construct start_input as a random tensor 
        # of shape (batch_size, input_len, embedding_dim) (embedding_dim = 768 for this GPT-2 model).
        start_input = normalise(start_input,[word_embeddings.min(dim=0)[0], word_embeddings.max(dim=0)[0]])
        # We normalise this random tensor so that its minimum and maximum values correspond to those in the entire word_embeddings tensor
        # This dispenses with whole swathes of "input space" which contain no legal token embeddings 
        # (we're limiting ourselves to a kind of "hull" defined by the 50527 vocab tokens in the embedding space), 
        # which is a sensible place to look for optimised inputs.
    else:

        _, centroids = kkmeans(word_embeddings, batch_size, seed=42)
        start_input = torch.cat([centroids.unsqueeze(1)]*input_len, dim=1)

    input = torch.nn.Parameter(start_input, requires_grad=True)
    # input is not a tensor, it's a Parameter object that wraps a tensor and adds additional functionality. 
    # 'input.data' is used below
    
    optimiser = torch.optim.Adam([input], lr=lr)
    # standard optimiser; note that it generally operates on a list of tensors, so we're giving it a list of one tensor; standard learning rate
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, 'min', patience=20, cooldown=20, factor=0.5)
    # this is used when loss hasn't improved for 20 timesteps; this scheduler will reduce the lr by a 'factor' of 0.5 when the 
    # validation loss stops improving for 'patience' (here 20) epochs, and will wait 'cooldown' (here 20) epochs before resuming normal operation.

    # now we loop across training epochs
    for e in range(epochs):
        logits, emb, perp = model_emb(model, torch.clamp(input, word_embeddings.min(), word_embeddings.max()), word_embeddings, output_len)
        # Does forward pass on a 'clamped' version of the 'input' tensor (done to contain it within the 'hull' of the vocabulary within 'input space').
        # Iterates to produce an output of output_len tokens, 
        # returns: 'logits' = tensor of logits for output, of shape (batch_size, output_len, vocab_size)
        # 'emb': tensor of embeddings for input+output of shape (batch_size, input_len + output_len, embedding_dim); 
        # 'perp': the input sequence perplexities tensor, of shape (batch_size,)
        probs = torch.softmax(logits, dim=-1)
        # For each batch, output, converts the sequence of logits (of length 'vocab_size') in the 'logits' tensor to probabilities, using softmax

        logits = (logits - logits.min(dim=-1)[0].unsqueeze(-1)) / (logits.max(dim=-1)[0].unsqueeze(-1) - logits.min(dim=-1)[0].unsqueeze(-1))
        # This appears to be normalising the logits for each batch/output embedding so they're all between 0 and 1... 
        # This is for ease of visualisation.

        perp_loss = perp.mean() 

        if output_len > output_ix.shape[0]:
            target_logits = logits[:,possible_target_positions,output_ix].max(dim=1)[0]
            target_probs = probs[:,possible_target_positions,output_ix].max(dim=1)[0]

        else:
            target_logits = logits[:,torch.arange(output_len), output_ix]
            target_probs = probs[:,torch.arange(output_len), output_ix]

            # This handles case where output_len == output_ix.shape[0]
            # target_logits now of shape (batch_size, output_len)
            # output_len < output_ix.shape[0] was dealt with in line 133
            
        
        token_dist, closest_ix = [],[]

        for b in input:
            tds, cixs = [], []
            for be in b:
                _, cix, td, _ = closest_tokens(be, word_embeddings, tokenizer)
                tds.append(td)
                cixs.append(cix)
            token_dist.append(torch.stack(tds))
            closest_ix.append(torch.stack(cixs))

        token_dist, closest_ix = torch.stack(token_dist).squeeze(-1), torch.stack(closest_ix).squeeze(-1)

        # As far as I can tell, this creates a tensor of shape (batch_size, input_len, 1) which gives distance to nearest
        # legal token embedding for each input embedding in each batch
        mean_token_dist = token_dist.mean() 

        # There are currently four loss types, many more could be introduced.
        # log_prob_loss is the current default.
        if loss_type == 'logit_loss':
            loss = 1-target_logits
        elif loss_type == 'log_prob_loss':
            loss = -torch.log(target_probs)
        elif loss_type == 'prob_loss':
            loss = 1-target_probs
        elif loss_type == 'CE':
            loss = torch.nn.functional.cross_entropy(logits.swapaxes(-1,-2), output_ix.repeat(batch_size, 1), reduction=None)
        else:
            print(loss_type + 'is not implemented.')
            return 

        batch_loss = loss.mean()

        total_loss = torch.stack([mean_token_dist * dist_reg, batch_loss, perp_loss * perp_reg]).mean()

        model_outs = model.module.generate(closest_ix, max_length = output_len+input_len)
        # The 'closest_ix' tensor is passed as the initial input sequence to the model, 
        # and the max_length parameter specifies the maximum length of the total sequence to generate.
        # The output sequence will be terminated either when the end-of-sequence token is generated 
        # or when the maximum length is reached, whichever occurs first.
        # 
        # The output of the model.generate method will be a tuple containing the generated sequences and the model's internal states. 
        # The generated sequences will be stored in a tensor of shape (batch_size, output_len+input_len). 
        # Each element of the tensor will be a sequence of tokens with a length of at most output_len+input_len.
        
        for b in range(batch_size):
            if target_output in tokenizer.decode(model_outs[b][input_len:]) and tokenizer.decode(model_outs[b]) not in optimised_inputs:
                
                done = tokenizer.decode(model_outs[b])
                optimised_inputs.add(done)
                metrics_table.add_data(*[tokenizer.decode(model_outs[b][:input_len]), tokenizer.decode(model_outs[b][input_len:])] + torch.stack([loss.squeeze(-1)[b], perp[b], token_dist.mean(dim=1)[b]], dim=-1).tolist())
                wandb.log({'Optimised Inputs': wandb.Html(''.join(['<p>{}.{}</p>'.format(i, repr(s)) for i, s in enumerate(optimised_inputs)]))})

                if rand_after:
                    input.data[b] = torch.rand_like(input[b])
                    # Random re-initialisation (if 'rand_after' set to True)
        
        if ((e+1) % w_freq == 0) or done and return_early:
        # Every w epochs we write to log, unless we have found an optimised input before that and 'return_early' == True. 
        # I'm still not entirely sure about the idea of 'return_early'.
             
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
                        # Change name to clarify (output of model if we just put in raw embeddings)
                        # prints batch number; closest_tokens(e)[0] is a list of tokens, closest_tokens(e)[0] is the first (closest) of these
                        # these get joined with separator '' (SHOULDN'T THAT BE ' '?)  
                    print(b, repr(' Closest embeddings: {}'.format(tokenizer.decode(model_outs[b]), '\n')))
                    closest_embeddings.append(tokenizer.decode(model_outs[b]))

            wandb.log({'Closest Embeddings': wandb.Html(''.join(['<p>{}.{}</p>'.format(i, repr(ce)) for i, ce in enumerate(closest_embeddings)])), 'Total Loss':total_loss, 'Mean Token Distance': mean_token_dist, 'Mean Loss': batch_loss, 'Mean Perplexity Loss':perp_loss, 'Epoch':e, 'LR':optimiser.param_groups[0]['lr'], 'Num Inputs Found':len(optimised_inputs)})

            if done and return_early:
                print('\nOptimised Input: "{}"'.format(done))
                return {'Metrics':metrics_table}
                # we know optimised_inputs set contains a single element in this case
            
        optimiser.zero_grad()
        total_loss.backward()
        optimiser.step()
        # I assume these three lines are standard NN optimisation stuff?

        if lr_decay:
            scheduler.step(total_loss)
         # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, 'min', patience=20, cooldown=20, factor=0.5) gets used if lr_decay == True
        done = None

    return {'Metrics':metrics_table}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb_user', type=str, default='jessicamarycooper')
    parser.add_argument('--model_name', type=str, default='gpt2')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--rand_after', action='store_true')
    parser.add_argument('--w_freq', type=int, default=10)
    parser.add_argument('--base_input', action='store_true')
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--input_len', type=int, default=3)
    parser.add_argument('--target_output', type=str, default='.')
    parser.add_argument('--output_len', type=int)
    parser.add_argument('--dist_reg', type=float, default=1)
    parser.add_argument('--perp_reg', type=float, default=0)
    parser.add_argument('--loss_type', type=str, default='log_prob_loss')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--return_early', action='store_true')
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--lr_decay', action='store_true')
    parser.add_argument('--note', type=str, default='')
    parser.add_argument('--run_test_set', action='store_true')
    parser.add_argument('--run_random', type=int, default=0)
    
    args = parser.parse_args()

    test_set = {" 4 5 6":3," D E F":3," a lot of data":3," the water":5," is that the government will":7," Jupiter, Saturn, Uranus,":7," np":4," y_1)":10," , quod est,":11}
    #test_set = [' 4 5 6', ' D E F', ' a lot of data', ' the water', ' is that the government will', ' Jupiter, Saturn, Uranus,', ' np', ' y_1)', ' , quod est,']

    torch.manual_seed(args.seed)

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('Using {} device.'.format(args.device))

    args.model, args.word_embeddings, args.tokenizer = load_all(args.model_name, args.device)
    args.model = torch.nn.DataParallel(args.model)

    if args.run_test_set:
        for to, il in test_set.items():
            args.input_len = il
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

    else:
        run = wandb.init(config=args, project='backwards', entity=args.wandb_user, reinit=True)
        results = optimise_input(**vars(args))
        wandb.log(results)
        run.finish()

    print(results)


