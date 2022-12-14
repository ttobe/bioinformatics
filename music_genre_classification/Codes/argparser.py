import parser
import argparse

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", help="batch_size", type=int, default=256)
    parser.add_argument("--num_node", help="number of hidden layer node", type=int, default=100)
    parser.add_argument("--num_hidden_layer", help="number of hidden layer", type=int, default=5)
    parser.add_argument("--epoch", help="number of hidden layer", type=int, default=50)
    parser.add_argument("--lr", help="learning rate", type=float, default=0.001)
    parser.add_argument("--dropout_rate", help="dropout_rate", type=float, default=0.3)
    args = parser.parse_args()
    params = vars(args)
    return params