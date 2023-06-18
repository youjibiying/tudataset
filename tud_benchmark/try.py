import auxiliarymethods.datasets as dp
from auxiliarymethods.gnn_evaluation import gnn_evaluation
from gnn_baselines.gnn_architectures import GIN,GOTN

from parser import *

args = get_parser()
print(args)

dataset = 'PROTEINS'
use_labels = True
# ["IMDB-BINARY", "REDDIT-BINARY", "PROTEINS",'ENZYMES'，'FRANKENSTEIN','COIL-RAG']
# Download dataset.
dp.get_dataset(dataset)
# gnn_type='gin'
gnn_type='gotn'
if gnn_type=='gin':
    model=GIN
elif gnn_type=='gotn':
    model=GOTN
args.gnn_type='got'

# Optimize the number of layers ({1,2,3,4,5} and
# the number of hidden features ({32,64,128}),
# set the maximum nummber of epochs to 200,
# batch size to 64,
# starting learning rate to 0.01, and
# number of repetitions for 10-CV to 10.
# print(gnn_evaluation(GIN, dataset, [1, 2, 3, 4, 5], [32, 64, 128], max_num_epochs=200,
#                      batch_size=64, start_lr=0.01, num_repetitions=10, all_std=True))
print(gnn_evaluation(model, dataset, [3,4,5], [32,64], max_num_epochs=200,
                     batch_size=32, start_lr=0.01, num_repetitions=1, all_std=True,args=args))