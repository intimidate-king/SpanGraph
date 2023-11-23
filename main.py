import argparse
from gnn.train import start_train, start_train_cv
from gnn.eval import eval_model_merge

import time
def get_args():
    parser = argparse.ArgumentParser(description='Model Implementation')
    parser.add_argument("--db_name", type=str, default='trainticket', help="db name")
    parser.add_argument("--dropout", type=float, default=0, help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1, help="gpu")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=500, help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=16, help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1, help="number of hidden gcn layers")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight for L2 loss")
    parser.add_argument("--r_train", type=float, default=0.7, help="training ratio")
    parser.add_argument("--r_test", type=float, default=0.2, help="testing ratio")
    parser.add_argument("--r_val", type=float, default=0.1, help="validation ratio")
    parser.add_argument("--model_name", type=str, default='trainticket', help="model_name")
    parser.add_argument("--fold", type=int, default=5, help="# of cv fold")
    args = parser.parse_args()
    return args

args = get_args()
print(args)

start_train(args)

#start_train_cv(args)

#eval_saved_model(args)

# eval_model_merge(args)

print('\ndone...')
