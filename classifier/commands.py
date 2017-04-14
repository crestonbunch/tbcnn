"""File for defining commands for the classifier."""

import argparse
import logging

import classifier.tbcnn.train as tbcnn_train
import classifier.tbcnn.test as tbcnn_test

def main():
    """Commands to train and test classifiers."""

    parser = argparse.ArgumentParser(
        description="Train and test classifiers on datasets.""",
    )
    subparsers = parser.add_subparsers(help='sub-command help')

    train_parser = subparsers.add_parser(
        'train', help='Train a model to classify'
    )
    train_parser.add_argument('model', type=str, help='Model to train: options are "tbcnn"')
    train_parser.add_argument('--infile', type=str, help='Data file to sample from')
    train_parser.add_argument('--logdir', type=str, help='File to store logs in')
    train_parser.add_argument(
        '--embedfile', type=str, help='Learned vector embeddings from the vectorizer'
    )
    train_parser.set_defaults(action='train')

    test_parser = subparsers.add_parser(
        'test', help='Test a model'
    )
    test_parser.add_argument('model', type=str, help='Model to train: options are "tbcnn"')
    test_parser.add_argument('--infile', type=str, help='Data file to sample from')
    test_parser.add_argument('--logdir', type=str, help='File to store logs in')
    test_parser.add_argument(
        '--embedfile', type=str, help='Learned vector embeddings from the vectorizer'
    )
    test_parser.set_defaults(action='test')

    args = parser.parse_args()

    if args.action == 'train':
        if args.model == 'tbcnn':
            tbcnn_train.train_model(args.logdir, args.infile, args.embedfile)

    if args.action == 'test':
        if args.model == 'tbcnn':
            tbcnn_test.test_model(args.logdir, args.infile, args.embedfile)
