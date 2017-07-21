# Copyright (c) Microsoft. All rights reserved.
#
# Written by Mary Wahl for a //oneweek tutorial

import numpy as np
import pandas as pd
import os, argparse, glob

def main(input_dir, output_dir, frac_test):
    # Load the filenames of all images
    df = pd.DataFrame([])
    df['filename'] = list(glob.iglob(os.path.join(input_dir, '*', '*')))
    df['label'] = df['filename'].apply(lambda x:
                                       os.path.basename(os.path.dirname(x)))

    # Record the mapping from labels to numerical indices
    labels = list(np.sort(df['label'].unique().tolist()))
    with open(os.path.join(output_dir, 'labels_to_inds.tsv'), 'w') as f:
        for i, label in enumerate(labels):
            f.write('{}\t{}\n'.format(label, i))
    df['idx'] = df['label'].apply(lambda x: labels.index(x))

    # Partition the data into test and training sets
    test_df = []
    train_df = []
    for label in labels:
        my_df = df.loc[df['label'] == label]
        proposed_n_test = int(len(my_df.index) * frac_test)
        proposed_n_training = len(my_df.index) - proposed_n_test
        if proposed_n_training < 1:
            raise Exception('''
Could not assign f={} for class {}: would have {} test and {} training samples
'''.format(frac_test, label, proposed_n_test, proposed_n_training))
        perm = np.random.permutation(my_df.index)
        test_df.append(my_df.loc[perm[:proposed_n_test]])
        train_df.append(my_df.loc[perm[proposed_n_test:]])
    test_df = pd.concat(test_df).sample(frac=1)
    train_df = pd.concat(train_df).sample(frac=1)

    # Write the MAP files
    with open(os.path.join(output_dir, 'map_train.tsv'), 'w') as f:
        for row in train_df.itertuples():
            f.write('{}\t{}\n'.format(row.filename, row.idx))
    with open(os.path.join(output_dir, 'map_test.tsv'), 'w') as f:
        for row in test_df.itertuples():
            f.write('{}\t{}\n'.format(row.filename, row.idx))
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''
Partitions dataset into training and testing, and creates a MAP file
describing each image's filename and label for each subset, as well as a text
file describing the mapping from label to index. Does not perform any class
rebalancing. The "input_dir" folder should contain only subfolders, and each
subfolder should contain only image files.''')
    parser.add_argument('-i', '--input_dir', type=str, required=True,
                        help='Directory containing all image files in ' +
                        'subfolders named by class (unpartitioned).')
    parser.add_argument('-o', '--output_dir',
                        type=str, required=True,
                        help='Output directory for output MAP and index-to' +
                        '-label text files.')
    parser.add_argument('-f', '--frac_test',
                        type=float, required=True,
                        help='Fraction of images in each class to reserve ' +
                        'for the test set (0 <= f < 1).')
    args = parser.parse_args()

    # Ensure specified files/directories exist
    assert os.path.exists(args.input_dir),
        'Input directory {} does not exist'.format(args.input_dir)
    assert (args.frac_test < 1.0) and (args.frac_test >= 0.0),
        'Require 0 <= f < 1'
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    main(args.input_dir, args.output_dir, args.frac_test)
