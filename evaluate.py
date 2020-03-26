import argparse
import os

import sys
from collections import defaultdict

import numpy as np
from mir_eval.multipitch import evaluate as evaluate_frames
from mir_eval.transcription import precision_recall_f1_overlap as evaluate_notes
from mir_eval.transcription_velocity import precision_recall_f1_overlap as evaluate_notes_with_velocity
from mir_eval.util import midi_to_hz
from scipy.stats import hmean

from onsets_and_frames import *

import tensorflow as tf
from tensorflow import keras

eps = sys.float_info.epsilon


def evaluate(metrics, model, inputs, targets, onset_threshold=0.5, frame_threshold=0.5, save_path=None):

    # NB: this can't be decorated with tf.function because of all the extract_notes functions not being pure TF code.

    mel = audio_to_mel(inputs)

    onset_pred, offset_pred, frame_pred, velocity_pred = model(mel, training=False)
    onset_labels, offset_labels, frame_labels, velocity_labels, path_labels = targets

    # for key, loss in losses.items():
    #     metrics[key].append(loss.item()) # todo: add loss metrics

    # We're working with batch size of 1, so remove the first index for everything.
    onset_pred = tf.squeeze(onset_pred)
    offset_pred = tf.squeeze(offset_pred)
    frame_pred = tf.squeeze(frame_pred)
    velocity_pred = tf.squeeze(velocity_pred)

    onset_labels = tf.squeeze(onset_labels)
    offset_labels = tf.squeeze(offset_labels)
    frame_labels = tf.squeeze(frame_labels)
    velocity_labels = tf.squeeze(velocity_labels)
    path_labels = tf.squeeze(path_labels).numpy().decode("utf-8")

    p_ref, i_ref, v_ref = extract_notes(onset_labels, frame_labels, velocity_labels)
    p_est, i_est, v_est = extract_notes(onset_pred, frame_pred, velocity_pred, onset_threshold, frame_threshold)

    t_ref, f_ref = notes_to_frames(p_ref, i_ref, frame_labels.shape)
    t_est, f_est = notes_to_frames(p_est, i_est, frame_pred.shape)

    scaling = HOP_LENGTH / SAMPLE_RATE

    i_ref = (i_ref * scaling).reshape(-1, 2)
    p_ref = np.array([midi_to_hz(MIN_MIDI + midi) for midi in p_ref])
    i_est = (i_est * scaling).reshape(-1, 2)
    p_est = np.array([midi_to_hz(MIN_MIDI + midi) for midi in p_est])

    t_ref = t_ref.astype(np.float64) * scaling
    f_ref = [np.array([midi_to_hz(MIN_MIDI + midi) for midi in freqs]) for freqs in f_ref]
    t_est = t_est.astype(np.float64) * scaling
    f_est = [np.array([midi_to_hz(MIN_MIDI + midi) for midi in freqs]) for freqs in f_est]

    p, r, f, o = evaluate_notes(i_ref, p_ref, i_est, p_est, offset_ratio=None)
    metrics['metric/note/precision'].append(p)
    metrics['metric/note/recall'].append(r)
    metrics['metric/note/f1'].append(f)
    metrics['metric/note/overlap'].append(o)

    p, r, f, o = evaluate_notes(i_ref, p_ref, i_est, p_est)
    metrics['metric/note-with-offsets/precision'].append(p)
    metrics['metric/note-with-offsets/recall'].append(r)
    metrics['metric/note-with-offsets/f1'].append(f)
    metrics['metric/note-with-offsets/overlap'].append(o)

    p, r, f, o = evaluate_notes_with_velocity(i_ref, p_ref, v_ref, i_est, p_est, v_est,
                                              offset_ratio=None, velocity_tolerance=0.1)
    metrics['metric/note-with-velocity/precision'].append(p)
    metrics['metric/note-with-velocity/recall'].append(r)
    metrics['metric/note-with-velocity/f1'].append(f)
    metrics['metric/note-with-velocity/overlap'].append(o)

    p, r, f, o = evaluate_notes_with_velocity(i_ref, p_ref, v_ref, i_est, p_est, v_est, velocity_tolerance=0.1)
    metrics['metric/note-with-offsets-and-velocity/precision'].append(p)
    metrics['metric/note-with-offsets-and-velocity/recall'].append(r)
    metrics['metric/note-with-offsets-and-velocity/f1'].append(f)
    metrics['metric/note-with-offsets-and-velocity/overlap'].append(o)

    frame_metrics = evaluate_frames(t_ref, f_ref, t_est, f_est)
    metrics['metric/frame/f1'].append(hmean([frame_metrics['Precision'] + eps, frame_metrics['Recall'] + eps]) - eps)

    for key, loss in frame_metrics.items():
        metrics['metric/frame/' + key.lower().replace(' ', '_')].append(loss)

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        label_path = os.path.join(save_path, os.path.basename(path_labels) + '.label.png')
        save_pianoroll(label_path, onset_labels, frame_labels)
        pred_path = os.path.join(save_path, os.path.basename(path_labels) + '.pred.png')
        save_pianoroll(pred_path, onset_pred, frame_pred)
        midi_path = os.path.join(save_path, os.path.basename(path_labels) + '.pred.mid')
        save_midi(midi_path, p_est, i_est, v_est)

    return metrics


def evaluate_file(checkpoint_dir, save_path, sequence_length, onset_threshold, frame_threshold):

    # Create default model and optimizer even though they'll be replaced with the checkpoint.
    model = OnsetsAndFrames(MAX_MIDI - MIN_MIDI + 1)
    optimizer = keras.optimizers.Adam(.0001)

    ckpt = tf.train.Checkpoint(step=tf.Variable(0), optimizer=optimizer, net=model)
    manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=3)

    ckpt.restore(manager.latest_checkpoint).expect_partial()
    if manager.latest_checkpoint:
        tf.print("Restored from {}".format(manager.latest_checkpoint))

    dataset = get_MAPS_Dataset('E:/data/MAPS', sequence_length=sequence_length)
    dataset = dataset.batch(1)

    metrics = defaultdict(list)
    for inputs, targets in dataset:
        metrics = evaluate(metrics, model, inputs, targets, onset_threshold, frame_threshold, save_path)

    for key, values in metrics.items():
        if key.startswith('metric/'):
            _, category, name = key.split('/')
            tf.print(f'{category:>32} {name:25}: {np.mean(values):.3f} ± {np.std(values):.3f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-dir', type=str, required=True, default=None)
    parser.add_argument('--save-path', type=str, required=False, default='evaluated',
                        help="If you want to save MIDI and piano roll images, specify a folder.")
    parser.add_argument('--sequence-length', default=SAMPLE_RATE*180, type=int) # todo: debug. some FLAC files might be too large for 8-GB GPUS
    parser.add_argument('--onset-threshold', default=0.5, type=float)
    parser.add_argument('--frame-threshold', default=0.5, type=float)

    evaluate_file(**vars(parser.parse_args()))
