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

os.environ["WANDB_API_KEY"] = "YOUR_WANDB_API_KEY"
os.environ["WANDB_SILENT"] = "true"


def optimise_input(model,
                   word_embeddings,
                   tokenizer,
                   device,
                   epochs=100,
                   lr=0.1,
                   no_reinit=False,
                   w_freq=10,  
                   rand_input=False,
                   local_input=False,
                   batch_size=20,
                   input_len=10,
                   target_output=' world',  
                   output_len=None,
                   dist_reg=0.1, 
                   perp_reg=0,
                   loss_type='log_prob_loss',
                   seed=0,
                   return_early=False,  # finishes if single optimised input is found
                   verbose=1,  
                   lr_decay=False,  # Use learning rate decay? If so, a scheduler gets invoked.
                   run_random=0,
                   equal_clusters=False,
                   penalise_repetition=False,
                   optimiser='Adam',
                   **kwargs):

  if run_random > 0:
        random_ix = (torch.rand(1) * word_embeddings.shape[0]).int()
        target_output = tokenizer.decode(random_ix)  # Converts token index to string representation
        wandb.config.update({'target_output': target_output}, allow_val_change=True)

    print('Optimising input of length {} to maximise output logits for "{}"'.format(input_len, target_output))
    done = None

    output_ix = tokenizer.encode(target_output, return_tensors='pt')[0].to(device)

    word_embeddings = word_embeddings / torch.sqrt(torch.sum(word_embeddings**2, dim=-1, keepdim=True))

    optimised_inputs = set()
    optimised_tokens = []
    metrics_table = wandb.Table(columns=['Input', 'Output', 'Loss', 'Perplexity', 'Distance', 'Probs'])

    if output_len == None or output_len < output_ix.shape[
        0]:  
        output_len = output_ix.shape[
            0]  
    else:
        possible_target_positions = torch.stack(
            [torch.arange(0, output_ix.shape[0]) + i for i in range(output_len - output_ix.shape[0] + 1)])

    if rand_input == True:
        start_input = word_embeddings[torch.randperm(word_embeddings.shape[0])[:input_len * batch_size]].reshape(
            batch_size, input_len, -1)
    elif local_input == True:
        local_embs = closest_tokens(word_embeddings[output_ix].mean(dim=0), word_embeddings, tokenizer, n=batch_size)[-1].unsqueeze(1)
        start_input = local_embs.repeat(1, input_len, 1)
    else:
        num_clusters = batch_size * input_len
        _, centroids = kkmeans(word_embeddings.detach(), num_clusters, seed=seed,
                               equal_clusters=equal_clusters)
        start_input = centroids.reshape(batch_size, input_len, -1)

    input = torch.nn.Parameter(start_input.to(device), requires_grad=True)

    if optimiser == 'Adam':
        optimiser = torch.optim.Adam([input], lr=lr, eps=0.0001)
    elif optimiser == 'SGD':
        optimiser = torch.optim.SGD([input], lr=lr)
    else:
        print('Unsupported optimiser: ', optimiser)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, 'min', patience=20, cooldown=20, factor=0.5)

    for e in range(epochs):
        norm_input = input / torch.sqrt(torch.sum(input**2, dim=-1, keepdim=True))
        logits, emb, perp = model_emb(model, norm_input,
                                      word_embeddings, output_len)

        probs = torch.softmax(logits, dim=-1)

        perp_loss = perp.mean()  # across all elements in the batch

        if output_len > output_ix.shape[0]:
            target_logits = logits[:, possible_target_positions, output_ix].max(dim=1)[0]
            target_probs = probs[:, possible_target_positions, output_ix].max(dim=1)[0]

        else:
            target_logits = logits[:, torch.arange(output_len), output_ix]
            target_probs = probs[:, torch.arange(output_len), output_ix]

        token_dist, closest_ix = [], []
        for b in norm_input:
            tds, cixs = [], []
            for be in b:
                _, cix, td, _ = closest_tokens(be, word_embeddings, tokenizer)
                tds.append(td)
                cixs.append(cix)
            token_dist.append(torch.stack(tds))
            closest_ix.append(torch.stack(cixs))

        token_dist, closest_ix = torch.stack(token_dist).squeeze(-1), torch.stack(closest_ix).squeeze(-1)

        mean_token_dist = token_dist.mean()

        if loss_type == 'log_prob_loss':
            loss = -torch.log(target_probs)
        elif loss_type == 'CE':
            if output_len > 1:
                print('CE not supported with output length > 1.')
                return
            loss = torch.nn.functional.cross_entropy(logits.swapaxes(-1, -2), output_ix.repeat(batch_size, 1),
                                                     reduction='none')
        else:
            print(loss_type + 'is not implemented.')
            return

        batch_loss = loss.mean()

        total_loss = torch.stack([mean_token_dist * dist_reg, batch_loss, perp_loss * perp_reg]).mean()

        if penalise_repetition:
            rep_penalty = logits[:, :input_len, output_ix].sum()
            total_loss += rep_penalty
        else:
            rep_penalty = 0

        model_outs = model.generate(closest_ix, max_length=output_len + input_len)
      
        for b in range(batch_size):
            if target_output in tokenizer.decode(model_outs[b][input_len:]):
                if tokenizer.decode(model_outs[b]) not in optimised_inputs:
                    optimised_tokens += [tokenizer.decode(t) for t in model_outs[b][:input_len]]

                    counts = Counter(optimised_tokens)
                    labels, values = zip(*counts.items())

                    data = [[label, val] for (label, val) in zip(labels, values)]
                    table = wandb.Table(data=data, columns=["Token", "Count"])
                    wandb.log({"token_freqs": wandb.plot.bar(table, "Token",
                                                             "Count", title="Token Freqs")})

                    done = tokenizer.decode(model_outs[b])
                    optimised_inputs.add(done)
                    metrics_table.add_data(*[tokenizer.decode(model_outs[b][:input_len]),
                                             tokenizer.decode(model_outs[b][input_len:])] + torch.stack(
                        [loss.squeeze(-1)[b].mean(), perp[b], token_dist.mean(dim=1)[b]], dim=-1).tolist() + [target_probs[
                                                b].tolist()])
                    wandb.log({'Optimised Inputs': wandb.Html(
                        ''.join(['<p>{}.{}</p>'.format(i, repr(s)) for i, s in enumerate(optimised_inputs)]))})

                if no_reinit == False:
                    if rand_input == True or local_input == True:
                        input.data[b] = word_embeddings[torch.randperm(word_embeddings.shape[0])[:input_len]].reshape(1,
                                                                                                                      input_len,
                                                                                                                      -1).to(
                            device)
                    else:
                        rand_centroids = centroids[np.random.randint(0, batch_size, size=input_len)].unsqueeze(0)
                        input.data[b] = rand_centroids

        if ((e + 1) % w_freq == 0) or done and return_early:

            print("Optimised Inputs:", optimised_inputs)
            print('{}/{} Output Loss: {} Emb Dist Loss: {} Perp Loss: {} LR: {}'.format(e + 1, epochs, batch_loss,
                                                                                        mean_token_dist, perp_loss,
                                                                                        optimiser.param_groups[0][
                                                                                            'lr']))
            if verbose == 3:
                print('Target Probs: {}\nTarget Logits: {}\nInput Dists: {}\nInput Perplexity: {}\n'.format(
                    target_probs.detach().cpu().numpy(), target_logits.detach().cpu().numpy(),
                    token_dist.detach().cpu().numpy(), perp.detach().reshape(-1).cpu().numpy()))

            closest_embeddings = []

            for b in range(batch_size):
                if verbose > 0:
                    if verbose == 2:
                        print(b, repr(' Raw embeddings: {}'.format(''.join([closest_tokens(e)[0][0] for e in emb[b]]))))

                    print(b, repr(' Closest embeddings: {}'.format(tokenizer.decode(model_outs[b]), '\n')))
                    closest_embeddings.append(tokenizer.decode(model_outs[b]))

            wandb.log({'Closest Embeddings': wandb.Html(
                ''.join(['<p>{}.{}</p>'.format(i, repr(ce)) for i, ce in enumerate(closest_embeddings)])),
                       'Total Loss': total_loss, 'Mean Token Distance': mean_token_dist, 'Mean Loss': batch_loss,
                       'Mean Perplexity Loss': perp_loss, 'Epoch': e, 'LR': optimiser.param_groups[0]['lr'],
                       'Num Inputs Found': len(optimised_inputs), 'Repetition Penalty': rep_penalty})

            if done and return_early:
                print('\nOptimised Input: "{}"'.format(done))
                return {'Metrics': metrics_table}

        optimiser.zero_grad()
        total_loss.backward()
        optimiser.step()

        if lr_decay:
            scheduler.step(total_loss)
        done = None

    return {'Metrics': metrics_table}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb_user', type=str, default='jessicamarycooper')
    parser.add_argument('--model_name', type=str, default='gpt2')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--no_reinit', action='store_true')
    parser.add_argument('--w_freq', type=int, default=10)
    parser.add_argument('--rand_input', action='store_true')
    parser.add_argument('--local_input', action='store_true')
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
    parser.add_argument('--optimiser', type=str, default='Adam')
    parser.add_argument('--equal_clusters', action='store_true')
    parser.add_argument('--penalise_repetition', action='store_true')

    args = parser.parse_args()

    test_sets = [
        [' externalToEVA', 'quickShip', ' TheNitrome', 'embedreportprint', 'rawdownload', 'reportprint', ' サーティ',
         ' RandomRedditor', 'oreAndOnline', 'InstoreAndOnline', ' externalTo', 'StreamerBot', 'ActionCode', 'Nitrome', ' SolidGoldMagikarp', 'PsyNetMessage'],
        [' girl', ' boy', 'good', ' evil', ' science', ' art', ' England', ' USA'],
        [' newcom', 'slaught', 'senal', 'imei']]

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

        seeds = (torch.rand(args.run_random) * 60000).int()
        for r in range(args.run_random):
            args.seed = seeds[r]
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

