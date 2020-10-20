"""
A rough translation of Magenta's Onsets and Frames implementation [1].

    [1] https://github.com/tensorflow/magenta/blob/master/magenta/models/onsets_frames_transcription/model.py
"""

from .lstm import BidirectionalLSTM

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv2D, InputLayer, Dropout, MaxPool2D, BatchNormalization, Flatten, Permute
from tensorflow.keras.layers import Reshape, ReLU

class ConvStack(keras.Sequential):
    def __init__(self, output_features, input_shape=None, **kwargs):
        # input is batch_size * 1 channel * frames * input_features

        super().__init__([
            InputLayer(input_shape=input_shape),
            # layer 0
            Conv2D(output_features // 16, 3, padding='same', data_format='channels_first'),
            BatchNormalization(),
            ReLU(),
            # layer 1
            Conv2D(output_features // 16, 3, padding='same', data_format='channels_first'),
            BatchNormalization(),
            ReLU(),
            # layer 2
            MaxPool2D((1, 2)),
            Dropout(0.25),
            Conv2D(output_features // 8, 3, padding='same', data_format='channels_first'),
            BatchNormalization(),
            ReLU(),
            # layer 3
            MaxPool2D((1, 2), data_format='channels_first'),
            Dropout(0.25),

            # https://github.com/jongwook/onsets-and-frames/blob/f5f5bc812a45d88f029452e52ad76ff742626ec3/onsets_and_frames/transcriber.py#L47
            Permute((2,1,3)), # swap the channel (filter count) with the frame index.

            # collapse the filter count and output_features, resulting in
            # a shape of (frames, (output_features // 8) * (input_shape[-1] // 4)))
            Reshape((-1, (output_features // 8) * (input_shape[-1] // 4))), # todo: very hard to understand this
            # https://github.com/jongwook/onsets-and-frames/blob/f5f5bc812a45d88f029452e52ad76ff742626ec3/onsets_and_frames/transcriber.py#L40

            # fully connected
            Dense(output_features),
            Dropout(0.5)
        ], **kwargs)


class OnsetsAndFrames(keras.models.Model):

    def __init__(self, num_pitch_classes, model_complexity=48, **kwargs):

        self.model_size = model_complexity * 16 # because ConvStack class needs a model size multiple of 16
        self.num_pitch_classes = num_pitch_classes

        super().__init__(**kwargs)

    def build(self, dims):

        # dims should be:
        # (batch size, 1 channel, num frames, num pitch classes)

        shape = (dims[-3], dims[-2], dims[-1])
        # shape should be:
        # (1 channel, num frames, num pitch classes)

        self.onset_stack = keras.Sequential([
            ConvStack(self.model_size, input_shape=shape),
            BidirectionalLSTM(self.model_size, self.model_size),
            Dense(self.num_pitch_classes, activation='sigmoid', name='onset'),
        ])

        self.offset_stack = keras.Sequential([
            ConvStack(self.model_size, input_shape=shape),
            BidirectionalLSTM(self.model_size, self.model_size),
            Dense(self.num_pitch_classes, activation='sigmoid', name='offset')
        ])

        self.frame_stack = keras.Sequential([
            ConvStack(self.model_size, input_shape=shape),
            Dense(self.num_pitch_classes, activation='sigmoid')
        ])

        self.combined_stack = keras.Sequential([
            BidirectionalLSTM(self.num_pitch_classes*3, self.model_size),
            Dense(self.num_pitch_classes, activation='sigmoid')
        ])

        self.velocity_stack = keras.Sequential([
            ConvStack(self.model_size, input_shape=shape),
            Dense(self.num_pitch_classes, activation=None)
        ])

        # super().build(dims) todo: can we call super?

    def call(self, inputs, training=False):

        mel = inputs

        onset_pred = self.onset_stack(mel)
        offset_pred = self.offset_stack(mel)
        activation_pred = self.frame_stack(mel)
        combined_pred = tf.concat([onset_pred, offset_pred, activation_pred], -1)
        frame_pred = self.combined_stack(combined_pred)
        velocity_pred = self.velocity_stack(mel)

        return onset_pred, offset_pred, frame_pred, velocity_pred
