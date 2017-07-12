# Copyright (c) Microsoft. All rights reserved.
#
# Written by Mary Wahl for a //oneweek tutorial

import numpy as np
import os, argparse, cntk

def create_minibatch_source(map_file, n_classes):
    transforms = [cntk.io.transforms.crop(crop_type='randomside',
                                          side_ratio=0.85,
                                          jitter_type='uniratio'),
                  cntk.io.transforms.scale(width=224,
                               height=224,
                               channels=3,
                               interpolations='linear'),
                  cntk.io.transforms.color(brightness_radius=0.2,
                               contrast_radius=0.2,
                               saturation_radius=0.2)]
    return(cntk.io.MinibatchSource(cntk.io.ImageDeserializer(map_file, cntk.io.StreamDefs(
        features = cntk.io.StreamDef(field='image', transforms=transforms, is_sparse=False),
        labels = cntk.io.StreamDef(field='label', shape=n_classes, is_sparse=False)))))


def load_pretrained_model(image_input, n_classes, pretrained_model_file):
    loaded_model = cntk.load_model(pretrained_model_file)
    feature_node = cntk.logging.graph.find_by_name(loaded_model, 'features')
    last_node = cntk.logging.graph.find_by_name(loaded_model, 'h2_d')
    all_layers = cntk.ops.combine([last_node.owner]).clone(cntk.ops.functions.CloneMethod.freeze,
                                                           {feature_node: cntk.ops.placeholder()})

    feat_norm = image_input - cntk.layers.Constant(114)
    fc_out = all_layers(feat_norm)
    new_model = cntk.layers.Dense(n_classes)(fc_out)
    return(new_model)


def main(input_map_file, output_retrained_model_file, pretrained_model_file):
    # Count the number of classes and samples in the MAP file
    labels = set([])
    epoch_size = 0
    with open(input_map_file, 'r') as f:
        for line in f:
            labels.add(line.strip().split('\t')[-1])
            epoch_size += 1
    n_classes = len(labels)

    # Create the training minibatch source
    minibatch_source = create_minibatch_source(input_map_file, n_classes)

    # Input variables for image and label data
    image_input = cntk.ops.input_variable((3, 224, 224))
    label_input = cntk.ops.input_variable((n_classes))
    input_map = {image_input: minibatch_source.streams.features,
                 label_input: minibatch_source.streams.labels}

    # Load the model file and modify as needed
    model = load_pretrained_model(image_input, n_classes, pretrained_model_file)

    # Set learning parameters
    ce = cntk.losses.cross_entropy_with_softmax(model, label_input)
    pe = cntk.metrics.classification_error(model, label_input)
    l2_reg_weight = 0.0005
    lr_per_sample = [0.00001] * 25 + [0.000001] * 15 + [0.0000001]
    momentum_time_constant = 10
    max_epochs = 50
    mb_size = 16
    lr_schedule = cntk.learners.learning_rate_schedule(lr_per_sample,
                                                       unit=cntk.UnitType.sample)
    mm_schedule = cntk.learners.momentum_as_time_constant_schedule(momentum_time_constant)

    # Instantiate the trainer object
    progress_writers = [cntk.logging.progress_print.ProgressPrinter(tag='Training',
                                                                    num_epochs=max_epochs)]
    learner = cntk.learners.fsadagrad(parameters=model.parameters,
                                      lr=lr_schedule,
                                      momentum=mm_schedule,
                                      l2_regularization_weight=l2_reg_weight)
    trainer = cntk.Trainer(model, (ce, pe), learner, progress_writers)

    # Get minibatches of images and perform model training
    print('Retraining AlexNet model for {} epochs.'.format(max_epochs))
    cntk.logging.progress_print.log_number_of_parameters(model)
    
    for epoch in range(max_epochs):
        sample_count = 0
        while sample_count < epoch_size:
            data = minibatch_source.next_minibatch(min(mb_size, epoch_size-sample_count),
                                                   input_map=input_map)
            trainer.train_minibatch(data)
            sample_count += trainer.previous_minibatch_sample_count
        trainer.summarize_training_progress()
    model.save(output_retrained_model_file)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''
Replaces and retrains the final layer of an AlexNet model. Expects a MAP file
describing the filename and label for each training image, and a pretrained
AlexNet model. Saves the retrained model to the specified output file.
''')
    parser.add_argument('-i', '--input_map_file', type=str, required=True,
                        help='Path to the MAP file describing training data')
    parser.add_argument('-o', '--output_retrained_model_file',
                        type=str, required=True,
                        help='Path where the retrained model will be saved.')
    parser.add_argument('-m', '--pretrained_model_file',
                        type=str, required=True,
                        help='Pretrained model file; download from here: ' +
                        'https://www.cntk.ai/Models/AlexNet/AlexNet.model')
    args = parser.parse_args()

    # Ensure specified files/directories exist
    for i in [args.input_map_file, args.pretrained_model_file]:
        assert os.path.exists(i), 'Input file {} does not exist'.format(i)
    if not os.path.exists(os.path.dirname(args.output_retrained_model_file)):
        os.makedirs(os.path.dirname(args.output_retrained_model_file))

    main(args.input_map_file, args.output_retrained_model_file, args.pretrained_model_file)