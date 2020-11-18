import os
import time
import argparse
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 2 will hide info/warning. 3 also hides errors.
import tensorflow as tf
tf.config.run_functions_eagerly(True)
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices: tf.config.experimental.set_memory_growth(device, True)
from tensorflow import keras
from tensorflow.python.client import device_lib
from onsets_and_frames import *

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch-size', type=int, default=8)  # should be 4 or 8
parser.add_argument('--validation-interval', type=int, default=5)
parser.add_argument('--model-complexity', type=int, default=48)
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

# the new checkpoint directory
checkpoint_dir = os.path.join(os.path.join(os.curdir, 'runs'), time.strftime("run_%Y_%m_%d-%H_%M_%S"))
os.makedirs(checkpoint_dir, exist_ok=True)

train_on = args.train_on
MAESTRO_FOLDER = args.maestro_folder
MAPS_FOLDER = args.maps_folder
batch_size = args.batch_size
model_complexity = args.model_complexity
clip_gradient_norm = args.clip_gradient
validation_interval = args.validation_interval
epochs = args.epochs
leave_one_out = args.leave_one_out
learning_rate = args.learning_rate
learning_rate_decay_steps = args.learning_rate_decay_steps
learning_rate_decay_rate = args.learning_rate_decay_rate

sequence_length = 327680

gpu_device_name = tf.test.gpu_device_name()  # '/device:GPU:0'

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
else:  # use MAPS

    dataset = get_MAPS_Dataset(MAPS_FOLDER, groups=['AkPnBcht', 'AkPnBsdf', 'AkPnCGdD', 'AkPnStgb', 'SptkBGAm',
                                                    'SptkBGCl', 'StbgTGd2'], sequence_length=sequence_length)
    validation_dataset = get_MAPS_Dataset(MAPS_FOLDER, groups=['ENSTDkAm', 'ENSTDkCl'],
                                          sequence_length=sequence_length)

dataset = dataset.batch(batch_size)
validation_dataset = validation_dataset.batch(batch_size)

lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    learning_rate,
    decay_steps=learning_rate_decay_steps,
    decay_rate=learning_rate_decay_rate,
    staircase=True)

model = OnsetsAndFrames(MAX_MIDI - MIN_MIDI + 1, model_complexity=model_complexity,
                        clip_gradient_norm=clip_gradient_norm)

tensorboard_cb = tf.keras.callbacks.TensorBoard(checkpoint_dir)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(checkpoint_dir, "best_val_total_loss.ckpt"),
                                                 verbose=0, save_best_only=True,
                                                 monitor='val_total_loss')
callbacks = [tensorboard_cb, cp_callback]

model.compile(keras.optimizers.Adam(learning_rate=lr_schedule))

history = model.fit(dataset, epochs=epochs, callbacks=callbacks, validation_data=validation_dataset,
                    validation_freq=validation_interval, initial_epoch=0)

tf.print('Finished training.')
