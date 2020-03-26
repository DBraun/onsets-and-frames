import os
import time

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 2 will hide info/warning. 3 also hides errors.

import tensorflow as tf
# tf.config.experimental_run_functions_eagerly(True)
# tf.config.experimental.set_memory_growth(gpu_devices[0], True)
from tensorflow import keras
from tensorflow.python.client import device_lib

from onsets_and_frames import *

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch-size', type=int, default=8) # should be 4 or 8
parser.add_argument('--checkpoint-dir', type=str, default=None,
                    help="Directory containing a checkpoint to restore. If unspecified, don't restore any checkpoint.")
parser.add_argument('--checkpoint-interval', type=int, default=1)
parser.add_argument('--validation-interval', type=int, default=1)
parser.add_argument('--model_complexity', type=int, default=48)
parser.add_argument('--clip-gradient', type=float, default=3)
parser.add_argument('--learning-rate', type=float, default=0.0006)
parser.add_argument('--learning-rate-decay-rate', type=float, default=.98)
parser.add_argument('--learning-rate-decay-steps', type=int, default=10000)
parser.add_argument('--train-on', type=str, default='MAESTRO', choices=['MAESTRO', 'MAPS'])
parser.add_argument('--leave-one-out', type=str, default=None, nargs='*',
                    choices=['2004', '2006', '2008', '2009', '2011', '2013', '2014', '2015', '2017', '2018'])
parser.add_argument('--maestro-folder', type=str, default='data/MAESTRO', help='The path to the MAESTRO dataset.')
parser.add_argument('--maps-folder', type=str, default='data/MAPS', help='The path to the MAPS dataset.')

args = parser.parse_args()

resume_iteration = False
if args.checkpoint_dir is not None:
    checkpoint_dir = args.checkpoint_dir
    resume_iteration = True
else:
    checkpoint_dir = os.path.join(os.path.join(os.curdir, 'runs'), time.strftime("run_%Y_%m_%d-%H_%M_%S"))
    os.makedirs(checkpoint_dir, exist_ok=True)

checkpoint_interval = args.checkpoint_interval
train_on = args.train_on
MAESTRO_FOLDER = args.maestro_folder
MAPS_FOLDER = args.maps_folder
batch_size = args.batch_size
model_complexity = args.model_complexity
clip_gradient_norm  = args.clip_gradient
validation_interval = args.validation_interval
epochs = args.epochs
leave_one_out = args.leave_one_out
learning_rate = args.learning_rate
learning_rate_decay_steps = args.learning_rate_decay_steps
learning_rate_decay_rate = args.learning_rate_decay_rate

sequence_length = 327680

gpu_device_name = tf.test.gpu_device_name() # '/device:GPU:0'

for device in device_lib.list_local_devices():
    if device.name == gpu_device_name and device.memory_limit < 10e9:
        batch_size //= 2
        sequence_length //= 2
        tf.print(f'Reducing batch size to {batch_size} and sequence_length to {sequence_length} to save memory')
        break

validation_length = sequence_length

if train_on == 'MAESTRO':

    if leave_one_out is not None:
        all_years = {'2004', '2006', '2008', '2009', '2011', '2013', '2014', '2015', '2017', '2018'}
        train_groups = list(all_years - {str(leave_one_out)})
        validation_groups = [str(leave_one_out)]
    else:
        train_groups, validation_groups = ['train'], ['validation']

    dataset = get_MAESTRO_Dataset(MAESTRO_FOLDER, groups=train_groups, sequence_length=sequence_length)
    validation_dataset = get_MAESTRO_Dataset(MAESTRO_FOLDER, groups=validation_groups,
                                             sequence_length=sequence_length)
else: # use MAPS

    dataset = get_MAPS_Dataset(MAPS_FOLDER, groups=['AkPnBcht', 'AkPnBsdf', 'AkPnCGdD', 'AkPnStgb', 'SptkBGAm',
                                                    'SptkBGCl', 'StbgTGd2'], sequence_length=sequence_length)
    validation_dataset = get_MAPS_Dataset(MAPS_FOLDER, groups=['ENSTDkAm', 'ENSTDkCl'],
                                          sequence_length=sequence_length)

dataset = dataset.batch(batch_size)

lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    learning_rate,
    decay_steps=learning_rate_decay_steps,
    decay_rate=learning_rate_decay_rate,
    staircase=True)
optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
model = OnsetsAndFrames(MAX_MIDI - MIN_MIDI + 1, model_complexity=model_complexity)

ckpt = tf.train.Checkpoint(step=tf.Variable(0), optimizer=optimizer, net=model)
manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=1)

if resume_iteration:
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        tf.print("Restored from {}".format(manager.latest_checkpoint))

# model.summary()

@tf.function
def grad(model, inputs, targets):
    with tf.GradientTape() as tape:

        mel = audio_to_mel(inputs)

        onset_pred, offset_pred, frame_pred, velocity_pred = model(mel, training=True)
        onset_labels, offset_labels, frame_labels, velocity_labels, path_labels = targets

        def velocity_loss(velocity_label, velocity_pred, onset_label):

            denominator = tf.reduce_sum(onset_label)
            if denominator == 0:
                return denominator
            else:
                return tf.reduce_sum((onset_label * (velocity_label - velocity_pred) ** 2)) / denominator

        loss_weights = {
            "onset": 1.0,
            "offset": 1.0,
            "frame": 1.0,
            "velocity": 1.0
        }

        loss_value = [
            loss_weights['onset'] * tf.keras.losses.BinaryCrossentropy()(onset_labels, onset_pred),
            loss_weights['offset'] * tf.keras.losses.BinaryCrossentropy()(offset_labels, offset_pred),
            loss_weights['frame'] * tf.keras.losses.BinaryCrossentropy()(frame_labels, frame_pred),
            loss_weights['velocity'] * velocity_loss(velocity_labels, velocity_pred, onset_labels)
        ]

        gradients = tape.gradient(loss_value, model.trainable_variables)

        loss_value = {
            'onset': loss_value[0],
            'offset': loss_value[1],
            'frame': loss_value[2],
            'velocity': loss_value[3]
        }

        return loss_value, gradients

class CustomLogger(object):

    def __init__(self, names, logdir):

        self._writer = tf.summary.create_file_writer(logdir)
        self._writer.set_as_default()

        self._epoch_loss_avgs = {}
        for name in names:
            self._epoch_loss_avgs[name] = tf.keras.metrics.Mean()

    def batch_loss(self, loss_values):
        for loss_name, loss_value in loss_values.items():
            self._epoch_loss_avgs[loss_name](loss_value)

    def end_epoch(self, epoch):
        epoch = int(epoch)
        tf.print('End Epoch: {0}'.format(epoch))
        for loss_name, loss_metric in self._epoch_loss_avgs.items():
            loss_value = loss_metric.result()
            loss_metric.reset_states()
            tf.summary.scalar(loss_name, loss_value, step=epoch)
            self._writer.flush()


custom_logger = CustomLogger(['onset','offset','frame','velocity'], checkpoint_dir)

first_epoch = int(ckpt.step)
for epoch in range(first_epoch, epochs):
    # https://www.tensorflow.org/tutorials/customization/custom_training_walkthrough

    # Training loop
    for inputs, targets in dataset:

        loss_values, grads = grad(model, inputs, targets)
        # tf.print(loss_values)
        clipped_gradients = [tf.clip_by_norm(g, clip_gradient_norm) for g in grads]
        optimizer.apply_gradients(zip(clipped_gradients, model.trainable_variables))

        custom_logger.batch_loss(loss_values) # Add current batch loss

    custom_logger.end_epoch(ckpt.step)

    ckpt.step.assign_add(1)

    if int(ckpt.step) % checkpoint_interval == 0:
        save_path = manager.save()
        print("Saved checkpoint for epoch {}: {}".format(int(ckpt.step), save_path))
