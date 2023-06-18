import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_parser():
    # Training settings
    parser = argparse.ArgumentParser(
        description='PyTorch implementation of pre-training of graph neural networks with funetune technique')
    parser.add_argument('--gpu', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--lr_scale', type=float, default=1,
                        help='relative learning rate for the feature extraction layer (default: 1)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--dataset', type=str, default='tox21',
                        help='root directory of dataset. For now, only classification.')
    parser.add_argument('--input_model_file', type=str, default='', help='filename to read the model (if there is any)')
    parser.add_argument('--filename', type=str, default='none', help='output filename')
    parser.add_argument('--seed', type=int, default=42, help="Seed for splitting the dataset.")
    parser.add_argument('--runseed', type=int, default=0, help="Seed for minibatch selection, random initialization.")
    parser.add_argument('--split', type=str, default="scaffold", help="random or scaffold or random_scaffold")
    parser.add_argument('--eval_train', type=int, default=0, help='evaluating training or not')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for dataset loading')
    parser.add_argument("--save_path", type=str,
                        default='/apdcephfs/share_1364275/jiyingzhang/models/finetune_gnns',
                        help="Where to save finetuned model.")

    # jizhi
    parser.add_argument('--jizhi', default=False, type=str2bool,
                        help='run in jizhi, automl control, do not need to open')
    # jiying for finetune
    parser.add_argument('--regularization_type', type=str,
                        # choices=['l2_sp', 'feature_map', 'attention_feature_map',"none"],
                        default='none', help='fine tune regularization.')
    parser.add_argument('--finetune_type', type=str,
                        default='none',
                        help='fine tune regularization.')  # choices=['delta', 'bitune', 'co_tune','l2_sp','none','bss'],
    parser.add_argument('--norm_type', type=str,
                        default='none', help='fine tune regularization.')
    parser.add_argument('--trade_off_backbone', default=0.0, type=float,
                        help='trade-off for backbone regularization')
    parser.add_argument('--trade_off_head', default=0.0, type=float,
                        help='trade-off for head regularization')
    ## bss
    parser.add_argument('--trade_off_bss', default=0.0, type=float,
                        help='trade-off for bss regularization')
    parser.add_argument('-k', '--k', default=1, type=int,
                        metavar='N',
                        help='hyper-parameter for BSS loss')
    parser.add_argument('--gtot_order', default=1, type=int, help='A^{k} in graph topology OT')

    # parameters for calculating channel attention
    parser.add_argument("--attention_file", type=str, default='channel_attention.pt',
                        help="Where to save and load channel attention file.")
    parser.add_argument("--data_path", type=str,
                        default='/apdcephfs/share_1364275/jiyingzhang/data/finetune_gnns/chem/dataset',
                        help="Where to save and load dataset.")

    parser.add_argument('--attention-batch-size', default=64, type=int,
                        metavar='N',
                        help='mini-batch size for calculating channel attention (default: 32)')
    parser.add_argument('--attention_epochs', default=50, type=int, metavar='N',
                        help='number of epochs to train for training before calculating channel weight')
    parser.add_argument('--attention-lr-decay-epochs', default=30, type=int, metavar='N',
                        help='epochs to decay lr for training before calculating channel weight')
    parser.add_argument('--attention_iteration_limit', default=50, type=int, metavar='N',
                        help='iteration limits for calculating channel attention, -1 means no limits')
    ## for stochnorm
    parser.add_argument('--prob', '--probability', default=0.5, type=float,
                        metavar='P', help='Probability for StochNorm layers')

    parser.add_argument('--print_freq', default=20, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--patience', type=int, default=20,
                        help='early stop patience.')

    parser.add_argument('--save_file', default='results.csv', help='save file name for results')
    parser.add_argument('--tag', default='none', help='tag for labeling the experiment')
    parser.add_argument('--debug', action='store_true', help='whether use the debug')


    ## for gtot
    parser.add_argument('--dropout_adj', default=False, type=str2bool, help="drop edge")
    parser.add_argument('--adaptive_weights', default=False, type=str2bool,
                        help="add the adaptive weights for reducing the effect of over-smoothing representation")
    parser.add_argument('--M', default=0.05, type=float,
                        help='min(x/M,1) for adaptive weights')
    parser.add_argument('--train_radio', default=1.0, type=float,
                        help='(train_set* train_radio) : val : test')

    parser.add_argument('--dist_metric', default='norm_cosine', type=str,
                        help='distance metric for optimal transport as cost matrix (cosine, norm_cosine)')

    ## for fgtot
    parser.add_argument('--Lambda', default=0.5, type=float,
                        help='Lambda * C + (1 - Lambda) * C_xy, it is the coefficient that we use to trade off node_cost matrix and GWD cost matrix')

    ## for graph level alignment
    parser.add_argument('--trade_off_gl', default=0.5, type=float,
                        help='it is the coefficient that we use to trade off graph-level OT')


    ## for dblp
    # pre-trained file
    parser.add_argument('--pre_trained_model_file', type=str, default='',
                        help='filename to read the model (if there is any)')
    parser.add_argument('--pool_trained_model_file', type=str, default='',
                        help='filename to read the model (if there is any)')
    parser.add_argument('--emb_trained_model_file', type=str, default='',
                        help='filename to read the model (if there is any)')
    parser.add_argument('--result_file', type=str, default='', help='output filename')
    # dataset settings
    parser.add_argument('--down_dataset', type=str, default='dblp',
                        help='downstream dataset name')
    parser.add_argument('--node_fea_dim', type=int, default=10,
                        help='node feature dimensions (BIO: 2; DBLP: 10))')
    parser.add_argument('--edge_fea_dim', type=int, default=1,
                        help='edge feature dimensions (BIO: 9; DBLP: 1))')


    args = parser.parse_args()

    # jizhi config check
    import json, os

    WORKSPACE_PATH = os.environ.get('JIZHI_WORKSPACE_PATH', '')
    if WORKSPACE_PATH:
        args = vars(args)  # 转化为dict

        text = open(WORKSPACE_PATH + '/job_param.json', 'r').read()
        jizhi_json = json.loads(text)
        for key in jizhi_json:
            args[key] = jizhi_json[key]
        # Open Jizhi Reporter
        args['jizhi'] = True
        args = argparse.Namespace(**args)  # 转为namespace
    else:
        args.jizhi = False

    return args
