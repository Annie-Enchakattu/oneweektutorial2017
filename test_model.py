# Copyright (c) Microsoft. All rights reserved.
#
# Written by Mary Wahl for a //oneweek tutorial

import numpy as np
import pandas as pd
import os, argparse, cntk, warnings
from PIL import Image

def load_image(filename):
    # load image data, ignoring any UserWarnings about bad metadata
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        image_data = Image.open(filename)

    # resize the image to the size AlexNet expects
    image_data = image_data.resize((224, 224), Image.BILINEAR)

    # make sure image is loaded in RGB format (not grayscale or RGBalpha)
    image_data = np.array(image_data.convert(mode='RGB'), dtype=np.float32)

    # flip from RGB order to BGR order of color channels
    image_data = image_data[:, :, ::-1]

    # change dim order from 224x224x3 to 3x224x224
    image_data = np.transpose(image_data, (2,0,1))

    # make the array contiguous to avoid a UserWarning about processing speed
    image_data = np.ascontiguousarray(image_data)

    return(image_data)

def print_class_specific_statistics(df, label):
    n_true_positive = len(df.loc[(df['true_label'] == label) &
                                 (df['pred_label'] == label)].index)
    n_false_positive = len(df.loc[(df['true_label'] != label) &
                                  (df['pred_label'] == label)].index)
    n_false_negative = len(df.loc[(df['true_label'] == label) &
                                  (df['pred_label'] != label)].index)
    n_true_negative = len(df.loc[(df['true_label'] != label) &
                                 (df['pred_label'] != label)].index)
    n_total = len(df.index)
    precision = n_true_positive / (n_true_positive + n_false_positive)
    recall = n_true_positive / (n_true_positive + n_false_negative)
    accuracy = (n_true_positive + n_true_negative) / len(df.index)
    print('''
For class {} ({} of {} samples):
- accuracy:  {:0.3f}
- precision: {:0.3f}
- recall:    {:0.3f}'''.format(label,
                               n_true_positive + n_false_negative,
                               n_total,
                               accuracy,
                               precision,
                               recall))
    return


def main(input_map_file, label_file, output_file, trained_model_file):
    model = cntk.load_model(trained_model_file)

    # Get the class names
    idx_to_label_dict = {}
    with open(label_file, 'r') as f:
        for line in f:
            label, idx = line.strip().split('\t')
            idx_to_label_dict[idx] = label
    n_classes = len(list(idx_to_label_dict.keys()))

    # Process each image and store true vs. predicted labels
    df = []
    with open(input_map_file, 'r') as f:
        for line in f:
            filename, label_idx = line.strip().split('\t')
            image_data = load_image(filename)
            raw_pred_result = model.eval({model.arguments[0]: image_data})
            pred_label_idx = np.argmax(np.squeeze(raw_pred_result))
            df.append([filename,
                       idx_to_label_dict[str(label_idx)],
                       idx_to_label_dict[str(pred_label_idx)]])
    df = pd.DataFrame(df, columns=['filename', 'true_label', 'pred_label'])
    accuracy = len(df.loc[df['true_label'] == df['pred_label']].index) / \
               len(df.index)
    print('Overall accuracy on test set: {:0.3f}'.format(accuracy))
    for label in idx_to_label_dict.values():
        print_class_specific_statistics(df, label)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''
Applies a trained model to images described in a MAP file. Prints overall
accuracy and metrics for each class. Requires a tab-delimited label-to-index
file to relate class indices back to class names. Optionally outputs
predictions to a CSV file.
''')
    parser.add_argument('-i', '--input_map_file', type=str, required=True,
                        help='Path to the MAP file describing test data.')
    parser.add_argument('-l', '--label_file', type=str, required=True,
                        help='Path to the label-to-index file.')
    parser.add_argument('-o', '--output_file',
                        type=str, required=False, default=None,
                        help='(Optional) Path for prediction results.')
    parser.add_argument('-m', '--trained_model_file',
                        type=str, required=True,
                        help='Retrained CNTK .model file.')
    args = parser.parse_args()

    # Ensure specified files/directories exist
    for i in [args.input_map_file, args.trained_model_file, args.label_file]:
        assert os.path.exists(i), 'Input file {} does not exist'.format(i)
    if args.output_file is not None:
        target_dir = os.path.dirname(args.output_file)
        if (len(target_dir) > 0) and not os.path.exists(target_dir):
            os.makedirs(target_dir)

    main(args.input_map_file, args.label_file, args.output_file,
         args.trained_model_file)