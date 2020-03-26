import tensorflow as tf

from .constants import *

@tf.function
def audio_to_mel(audio):

    stfts = tf.signal.stft(audio, WINDOW_LENGTH, HOP_LENGTH, fft_length=WINDOW_LENGTH, window_fn=tf.signal.hann_window,
                           pad_end=True)
    spectrograms = tf.abs(stfts)

    # Warp the linear scale spectrograms into the mel-scale.
    num_spectrogram_bins = stfts.shape[-1]

    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(N_MELS, num_spectrogram_bins, SAMPLE_RATE,
                                                                        MEL_FMIN, MEL_FMAX)

    mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)
    mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))

    # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
    log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)

    # reshape into a 4-dimensional batch where the first index will be the batch size.
    newshape = (log_mel_spectrograms.shape[-3],
                1,
                log_mel_spectrograms.shape[-2],
                log_mel_spectrograms.shape[-1])

    return tf.reshape(log_mel_spectrograms, newshape)
