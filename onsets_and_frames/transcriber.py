"""
A rough translation of Magenta's Onsets and Frames implementation [1].

    [1] https://github.com/tensorflow/magenta/blob/master/magenta/models/onsets_frames_transcription/model.py
"""

from .lstm import BidirectionalLSTM

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv2D, InputLayer, Dropout, MaxPool2D, BatchNormalization, Permute,\
    Reshape, ReLU
from onsets_and_frames.mel import audio_to_mel


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
            Permute((2, 1, 3)),  # swap the channel (filter count) with the frame index.

            # collapse the filter count and output_features, resulting in
            # a shape of (frames, (output_features // 8) * (input_shape[-1] // 4)))
            Reshape((-1, (output_features // 8) * (input_shape[-1] // 4))),  # todo: very hard to understand this
            # https://github.com/jongwook/onsets-and-frames/blob/f5f5bc812a45d88f029452e52ad76ff742626ec3/onsets_and_frames/transcriber.py#L40

            # fully connected
            Dense(output_features),
            Dropout(0.5)
        ], **kwargs)


class OnsetsAndFrames(keras.models.Model):

    def __init__(self, num_pitch_classes, model_complexity=48, clip_gradient_norm=3., **kwargs):

        super(OnsetsAndFrames, self).__init__(**kwargs)

        self.model_size = model_complexity * 16 # because ConvStack class needs a model size multiple of 16
        self.num_pitch_classes = num_pitch_classes
        self.clip_gradient_norm = clip_gradient_norm

        self._metric_loss_onset = tf.keras.metrics.Mean(name='onset')
        self._metric_loss_offset = tf.keras.metrics.Mean(name='offset')
        self._metric_loss_frame = tf.keras.metrics.Mean(name='frame')
        self._metric_loss_velocity = tf.keras.metrics.Mean(name='velocity')
        self._metric_loss_total = tf.keras.metrics.Mean(name='total_loss')

    def build(self, dims):

        # dims should be:
        # (batch size, 1 channel, num frames, num pitch classes)

        shape = (dims[-3], dims[-2], dims[-1])
        # shape should be:
        # (1 channel, num frames, num pitch classes)

        self.onset_stack = keras.Sequential([
            ConvStack(self.model_size, input_shape=shape),
            BidirectionalLSTM(self.model_size, self.model_size),
            Dense(self.num_pitch_classes, activation='sigmoid'),
        ], name='onset')

        self.offset_stack = keras.Sequential([
            ConvStack(self.model_size, input_shape=shape),
            BidirectionalLSTM(self.model_size, self.model_size),
            Dense(self.num_pitch_classes, activation='sigmoid')
        ], name='offset')

        self.frame_stack = keras.Sequential([
            ConvStack(self.model_size, input_shape=shape),
            Dense(self.num_pitch_classes, activation='sigmoid')
        ], name='frame')

        self.combined_stack = keras.Sequential([
            BidirectionalLSTM(self.num_pitch_classes*3, self.model_size),
            Dense(self.num_pitch_classes, activation='sigmoid')
        ])

        self.velocity_stack = keras.Sequential([
            ConvStack(self.model_size, input_shape=shape),
            Dense(self.num_pitch_classes, activation=None)
        ], name='velocity')

        # super().build(dims) todo: can we call super?

    def call(self, inputs, training=None, mask=None):

        mel = inputs

        onset_pred = self.onset_stack(mel)
        offset_pred = self.offset_stack(mel)
        activation_pred = self.frame_stack(mel)
        combined_pred = tf.concat([onset_pred, offset_pred, activation_pred], -1)
        frame_pred = self.combined_stack(combined_pred)
        velocity_pred = self.velocity_stack(mel)

        return onset_pred, offset_pred, frame_pred, velocity_pred

    @staticmethod
    def velocity_loss(velocity_label, velocity_pred, onset_label):

        denominator = tf.reduce_sum(onset_label)
        if denominator == 0:
            return denominator
        else:
            diff = velocity_label - velocity_pred
            return tf.reduce_sum((onset_label * diff*diff)) / denominator

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [self._metric_loss_onset, self._metric_loss_offset, self._metric_loss_frame,
                self._metric_loss_velocity, self._metric_loss_total]

    def get_loss(self, data, training=None):

        inputs, targets = data
        mel = audio_to_mel(inputs)

        onset_pred, offset_pred, frame_pred, velocity_pred = self(mel, training=training)
        onset_labels, offset_labels, frame_labels, velocity_labels, path_labels = targets

        loss_weights = {
            "onset": 1.0,
            "offset": 1.0,
            "frame": 1.0,
            "velocity": 1.0
        }

        losses = {
            'onset': loss_weights['onset'] * tf.keras.losses.BinaryCrossentropy()(onset_labels, onset_pred),
            'offset': loss_weights['offset'] * tf.keras.losses.BinaryCrossentropy()(offset_labels, offset_pred),
            'frame': loss_weights['frame'] * tf.keras.losses.BinaryCrossentropy()(frame_labels, frame_pred),
            'velocity': loss_weights['velocity'] * self.velocity_loss(velocity_labels, velocity_pred, onset_labels)
        }
        return losses

    def compute_metrics(self, loss_value):

        self._metric_loss_onset.update_state(loss_value['onset'])
        self._metric_loss_offset.update_state(loss_value['offset'])
        self._metric_loss_frame.update_state(loss_value['frame'])
        self._metric_loss_velocity.update_state(loss_value['velocity'])
        self._metric_loss_total.update_state(sum(loss_value.values()))

        return {m.name: m.result() for m in self.metrics}

    def train_step(self, data):

        with tf.GradientTape() as tape:

            loss_value = self.get_loss(data, training=True)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss_value, trainable_vars)

        clipped_gradients = [tf.clip_by_norm(g, self.clip_gradient_norm) for g in gradients]
        self.optimizer.apply_gradients(zip(clipped_gradients, trainable_vars))

        # Compute our metrics
        return self.compute_metrics(loss_value)

    def test_step(self, data):

        loss_value = self.get_loss(data)

        # Compute our metrics
        return self.compute_metrics(loss_value)

    def predict_step(self, data):

        mel = audio_to_mel(data)

        return self(mel, training=False)

    def dumb_predict(self, sequence_length):

        audio = tf.zeros((1, 1, sequence_length))

        return self.predict(audio)
