import sys, argparse, pickle, os, random
from importlib import import_module
import torch
import torch.nn as nn
import numpy as np
import math

from decoder import predictors, decoders
#import adaptive_softmax.model as asmodel

path = os.path.realpath(__file__)
path = path[:path.rindex('/')+1]
sys.path.insert(0, os.path.join(path, 'word_rep/'))
sys.path.insert(0, os.path.join(path, 'lm/'))
sys.path.insert(0, os.path.join(path, 'utils/'))
sys.path.insert(0, os.path.join(path, 'entailment/'))
sys.path.insert(0, os.path.join(path, 'context/'))
sys.path.insert(0, os.path.join(path, 'word_level/'))
sys.path.insert(0, os.path.join(path, 'diction/'))
sys.path.insert(0, os.path.join(path, 'reprnn/'))
sys.path.insert(0, os.path.join(path, 'style/'))
sys.path.insert(0, os.path.join(path, 'adaptive_softmax/'))

from adaptive_softmax import AdaptiveLoss

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='input.txt',
                    help='text file containing initial strings to continue')
parser.add_argument('--out', type=str, default='output.txt',
                    help='text file to write generations to')
parser.add_argument('--lm', type=str, default='lm.pt',
                    help='lm to use for decoding')
parser.add_argument('--dic', type=str, default='dic.pickle',
                    help='dic to use for lm')
parser.add_argument('--cutoffs', nargs='+', type=int,
                    help='cutoffs for buckets in adaptive softmax')
parser.add_argument('--verbosity', type=int, default=0,
                    help='how verbose to be during decoding')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--batch_size', type=int, default=1,
                    help='batch size')
## Decoding Stuff
parser.add_argument('--beam_size', type=int, default=10,
                    help='number of candidates in beam at every step')
parser.add_argument('--term', type=str, default='<end>',
                    help='what string to use as the end token')
parser.add_argument('--sep', type=str, default='</s>',
                    help='what string to use as the sentence seperator token')
parser.add_argument('--temp', type=float, default=None,
                    help='temperature, if using stochastic decoding')
parser.add_argument('--ranking_loss', action='store_true',
                    help='metaweight learning ranking loss')
parser.add_argument('--paragraph_level_score', action='store_true',
                    help='paragraph level score')
# Arbitrary Scorers
args = parser.parse_args()

np.random.seed(args.seed)

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

print("Load model")
with open(args.lm, 'rb') as model_file:
    model = torch.load(model_file)

model.eval()
with open(args.dic, 'rb') as dic_file:
    dictionary = pickle.load(dic_file)
ntokens = len(dictionary)

###############################################################################
# Build the model
###############################################################################

cutoffs = args.cutoffs + [ntokens]
criterion = AdaptiveLoss(cutoffs)

def evaluate(data_source, data_target, batch_size=10):
    # Turn on evaluation mode which disables dropout.
    hidden = model.init_hidden(batch_size)
    model.softmax.set_target(data_target.data.view(-1))
    output, hidden = model(data_source, hidden)
    #output_flat = output.view(-1, ntokens)
    loss = criterion(output, data_target.view(-1)).data.sum()
    return loss 


###############################################################################
# Load data
###############################################################################

with open(args.data) as data_file:
    val_loss = 0
    with torch.no_grad():
      for i, line in enumerate(data_file):
        init_tokens = line.strip().lower().split()
        init_tokens_ints = [dictionary[token] for token in init_tokens]
        strip_len = len(init_tokens) // args.batch_size 
        usable = strip_len * args.batch_size 
        data = np.asarray(init_tokens_ints[:usable]).reshape(args.batch_size, strip_len).transpose()

        source = torch.LongTensor(data[:-1, :]) 
        target = torch.LongTensor(data[1:, :]) 
        if args.cuda:
            source = source.cuda()
            target = target.cuda()
        #source = Variable(source, volatile=True)
        #target = Variable(target)
        loss = evaluate(source, target, args.batch_size)
        print('| instance loss {:5.2f} | instance ppl {:8.2f}'.format(
            loss, math.exp(loss)))
        val_loss += loss
    val_loss /= i

# Run on val data.
#val_loss = evaluate(val_data, test_batch_size)
print('=' * 89)
print('| End of pointer | val loss {:5.2f} | val ppl {:8.2f}'.format(
    val_loss, math.exp(val_loss)))
print('=' * 89)

