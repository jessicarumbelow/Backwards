import argparse
import torch
from helpers import *

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, default='gpt2')
parser.add_argument('-i', '--input', type=str)
parser.add_argument('-o', '--output_length', type=int)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model, embeddings, tokenizer = load_all(args.model, device)

ix = tokenizer.encode(args.input)

print('{} input tokens: {}'.format(len(ix), [tokenizer.decode(i) for i in ix]))

model_out = model.generate(torch.tensor(ix).unsqueeze(0).to(device), max_length = args.output_length + len(ix))

print('\nOutput:\n{}'.format(tokenizer.decode(model_out[0])))
