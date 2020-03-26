import json
import os

from glob import glob

import numpy as np
import soundfile

from .constants import *
from .midi import parse_midi
import tensorflow as tf
import tensorlayer

def maps_file_iterable(rootpath, groups=None):

    valid_groups = ['ENSTDkAm', 'ENSTDkCl'] if groups is None else groups

    # ['AkPnBcht', 'AkPnBsdf', 'AkPnCGdD', 'AkPnStgb', 'ENSTDkAm', 'ENSTDkCl', 'SptkBGAm', 'SptkBGCl', 'StbgTGd2']

    results = []
    for group in valid_groups:

        flacs = glob(os.path.join(rootpath, 'flac', '*_%s.flac' % group))
        if '\\flac\\' in flacs[0]:
            tsvs = [f.replace('\\flac\\', '\\tsv\\matched\\').replace('.flac', '.tsv') for f in flacs]
        else:
            tsvs = [f.replace('/flac/', '/tsv/matched/').replace('.flac', '.tsv') for f in flacs]

        assert(all(os.path.isfile(flac) for flac in flacs))
        assert(all(os.path.isfile(tsv) for tsv in tsvs))

        for audio_path, tsv_filename in sorted(zip(flacs, tsvs)):
            results.append([audio_path, tsv_filename])

    return results


def maestro_file_iterable(roothpath, groups=None):

    valid_groups = ['train'] if groups is None else groups

    results = []
    for group in valid_groups:

        if group not in ['train', 'validation', 'test']:
            # year-based grouping
            flacs = sorted(glob(os.path.join(roothpath, group, '*.flac')))
            if len(flacs) == 0:
                flacs = sorted(glob(os.path.join(roothpath, group, '*.wav')))

            midis = sorted(glob(os.path.join(roothpath, group, '*.midi')))
            files = list(zip(flacs, midis))
            if len(files) == 0:
                raise RuntimeError(f'Group {group} is empty')
        else:
            metadata = json.load(open(os.path.join(roothpath, 'maestro-v2.0.0.json')))
            files = sorted([(os.path.join(roothpath, row['audio_filename'].replace('.wav', '.flac')),
                             os.path.join(roothpath, row['midi_filename'])) for row in metadata if row['split'] == group])

            files = [(audio if os.path.exists(audio) else audio.replace('.flac', '.wav'), midi) for audio, midi in files]

        for audio_path, midi_path in files:
            tsv_filename = midi_path.replace('.midi', '.tsv').replace('.mid', '.tsv')
            if not os.path.exists(tsv_filename):
                midi = parse_midi(midi_path)
                np.savetxt(tsv_filename, midi, fmt='%.6f', delimiter='\t', header='onset,offset,note,velocity')
            results.append([audio_path, tsv_filename])

    return results


def load_audio_and_tsv(audio_path, tsv_path):
    """
    load an audio track and the corresponding labels

    Returns
    -------
        A dictionary containing the following data:

        path: str
            the path to the audio file

        audio: tensorflow.Tensor, shape = [num_samples]
            the raw waveform, not a batch.

        label: tensorflow.Tensor, shape = [num_steps, midi_bins]
            a matrix that contains the onset/offset/frame labels encoded as:
            3 = onset, 2 = frames after onset, 1 = offset, 0 = all else

        velocity: tensorflow.Tensor, shape = [num_steps, midi_bins]
            a matrix that contains MIDI velocity values at the frame locations
    """
    saved_data_path = audio_path.replace('.flac', '.npy').replace('.wav', '.npy')
    if os.path.exists(saved_data_path):
        return tensorlayer.files.load_npy_to_any(name=saved_data_path)

    audio, sr = soundfile.read(audio_path, dtype='int16')
    assert(sr == SAMPLE_RATE)

    audio = tf.convert_to_tensor(audio, dtype=tf.int16)
    audio_length = len(audio)

    n_keys = MAX_MIDI - MIN_MIDI + 1
    n_steps = (audio_length - 1) // HOP_LENGTH + 1

    label = np.zeros((n_steps, n_keys), dtype=np.uint8)
    velocity = np.zeros((n_steps, n_keys), dtype=np.uint8)

    midi = np.loadtxt(tsv_path, delimiter='\t', skiprows=1)

    for onset, offset, note, vel in midi:
        left = int(round(onset * SAMPLE_RATE / HOP_LENGTH))
        onset_right = min(n_steps, left + HOPS_IN_ONSET)
        frame_right = int(round(offset * SAMPLE_RATE / HOP_LENGTH))
        frame_right = min(n_steps, frame_right)
        offset_right = min(n_steps, frame_right + HOPS_IN_OFFSET)

        f = int(note) - MIN_MIDI
        label[left:onset_right, f] = 3
        label[onset_right:frame_right, f] = 2
        label[frame_right:offset_right, f] = 1
        velocity[left:frame_right, f] = vel

    label = tf.convert_to_tensor(label, dtype=tf.uint8)
    velocity = tf.convert_to_tensor(velocity, dtype=tf.uint8)

    data = dict(path=audio_path, audio=audio, label=label, velocity=velocity)

    tensorlayer.files.save_any_to_npy(save_dict=data, name=saved_data_path)
    # return tensorlayer.files.load_npy_to_any(name=saved_data_path) # for debugging to make sure loading works.
    return data


def make_post_process(sequence_length, seed):

    randomState = np.random.RandomState(seed) # todo: weird variable pollution idea

    def generator(data):

        result = dict(path=data['path'])
        audio_path = data['path']

        if sequence_length is not None:
            audio_length = len(data['audio'])
            seq_length = min(sequence_length, audio_length)

            if audio_length - seq_length > 0:
                # randint doesn't take 0 as a value.
                step_begin = randomState.randint(audio_length - seq_length) // HOP_LENGTH
            else:
                step_begin = 0

            n_steps = seq_length // HOP_LENGTH
            step_end = step_begin + n_steps

            begin = step_begin * HOP_LENGTH
            end = begin + seq_length

            result['audio'] = data['audio'][begin:end]
            result['label'] = data['label'][step_begin:step_end, :]
            result['velocity'] = data['velocity'][step_begin:step_end, :]
        else:
            result['audio'] = data['audio']
            result['label'] = data['label']
            result['velocity'] = data['velocity']

        audio = tf.cast(result['audio'], dtype=tf.float32) / 32768.0
        onset = tf.cast(result['label'] == 3, dtype=tf.float32)
        offset = tf.cast(result['label'] == 1, dtype=tf.float32)
        frame = tf.cast(result['label'] > 1, dtype=tf.float32)
        velocity = tf.cast(result['velocity'], dtype=tf.float32) / 128.0

        features = audio
        labels = (onset, offset, frame, velocity, audio_path)

        return features, labels

    return generator


def make_generator(file_iterator, post_process):

    def generator():

        for audio_path, tsv_path in file_iterator:
            features, labels = post_process(load_audio_and_tsv(audio_path, tsv_path))
            yield features, labels

    return generator


def get_Dataset(file_iterable, post_process):

    generator = make_generator(file_iterable, post_process)
    dataset = tf.data.Dataset.from_generator(generator, (tf.float32,
                                                         (tf.float32, tf.float32, tf.float32, tf.float32, tf.string)))
    return dataset


def get_MAESTRO_Dataset(rootpath, groups=None, sequence_length=None):

    file_iterable = maestro_file_iterable(rootpath, groups=groups)
    post_process = make_post_process(sequence_length, 42)

    return get_Dataset(file_iterable, post_process)


def get_MAPS_Dataset(rootpath, groups=None, sequence_length=None):

    file_iterable = maps_file_iterable(rootpath, groups=groups)
    post_process = make_post_process(sequence_length, 42)

    return get_Dataset(file_iterable, post_process)
