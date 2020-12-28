import argparse
import os
import sys
import glob

import numpy as np
import soundfile
from mir_eval.util import midi_to_hz

from onsets_and_frames import *
import tensorflow as tf


def load_and_process_audio(flac_path, sequence_length):

    random = np.random.RandomState(seed=42)

    audio, sr = soundfile.read(flac_path, dtype='int16')
    assert(sr == SAMPLE_RATE)

    audio = tf.convert_to_tensor(audio, dtype=tf.int16)
    audio = tf.cast(audio, dtype=tf.float32) / 32768.0

    if sequence_length is not None:
        audio_length = len(audio)
        step_begin = random.randint(audio_length - sequence_length) // HOP_LENGTH
        # n_steps = sequence_length // HOP_LENGTH

        begin = step_begin * HOP_LENGTH
        end = begin + sequence_length

        audio = audio[begin:end]

    return audio


def transcribe(model, audio):

    mel = audio_to_mel(audio)

    onset_pred, offset_pred, frame_pred, velocity_pred = model(mel, training=False)

    # reshape to remove the batch index. tf.squeeze can do this too.
    onset_pred = tf.reshape(onset_pred, (onset_pred.shape[1], onset_pred.shape[2]))
    offset_pred = tf.reshape(offset_pred, (offset_pred.shape[1], offset_pred.shape[2]))
    frame_pred = tf.reshape(frame_pred, (frame_pred.shape[1], frame_pred.shape[2]))
    velocity_pred = tf.reshape(velocity_pred, (velocity_pred.shape[1], velocity_pred.shape[2]))

    predictions = {
        'onset': onset_pred,
        'offset': offset_pred,
        'frame': frame_pred,
        'velocity': velocity_pred
    }

    return predictions


def transcribe_file(checkpoint_dir, model_complexity, flac_paths, save_path, sequence_length,
                    onset_threshold, frame_threshold):

    # Create default model and optimizer even though they'll be replaced with the checkpoint.
    model = OnsetsAndFrames(num_pitch_classes=MAX_MIDI - MIN_MIDI + 1, model_complexity=model_complexity)
    model.dumb_predict(sequence_length)  # We need to run the model on some data before we can load weights.
    model.load_weights(os.path.join(os.path.abspath(checkpoint_dir), 'best_val_total_loss.ckpt'))

    globbed_paths = glob.glob(flac_paths)

    for flac_path in globbed_paths:
        print(f'Processing FLAC: {flac_path}', file=sys.stderr)
        audio = load_and_process_audio(flac_path, sequence_length)

        audio = tf.expand_dims(audio, 0)

        predictions = transcribe(model, audio)

        p_est, i_est, v_est = extract_notes(predictions['onset'], predictions['frame'], predictions['velocity'],
                                            onset_threshold, frame_threshold)

        scaling = HOP_LENGTH / SAMPLE_RATE

        i_est = (i_est * scaling).reshape((-1, 2))
        p_est = np.array([midi_to_hz(MIN_MIDI + midi) for midi in p_est])

        os.makedirs(save_path, exist_ok=True)

        midi_path = os.path.join(save_path, os.path.basename(flac_path) + '.pred.mid')
        save_midi(midi_path, p_est, i_est, v_est)
        pred_path = os.path.join(save_path, os.path.basename(flac_path) + '.pred.png')
        save_pianoroll(pred_path, predictions['onset'], predictions['frame'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-dir', type=str, required=True, default=None)
    parser.add_argument('--model-complexity', type=int, default=48)
    parser.add_argument('--flac-paths', type=str, required=True, default='glob/path/for/*.flac',
                        help='A glob* expression for finding FLAC files.')
    parser.add_argument('--save-path', type=str, required=True, default='evaluated',
                        help='Directory for saving MIDI and piano roll PNG files.')
    parser.add_argument('--sequence-length', default=327680, type=int, help='Trim audio to this number of samples.')
    parser.add_argument('--onset-threshold', default=0.5, type=float)
    parser.add_argument('--frame-threshold', default=0.5, type=float)

    transcribe_file(**vars(parser.parse_args()))
