import numpy as np


def extract_notes(onsets, frames, velocity, onset_threshold=0.5, frame_threshold=0.5):
    """
    Finds the note timings based on the onsets and frames information

    Parameters
    ----------
    onsets: tensorflow.Tensor, shape = [frames, bins]
    frames: tensorflow.Tensor, shape = [frames, bins]
    velocity: tensorflow.Tensor, shape = [frames, bins]
    onset_threshold: float
    frame_threshold: float

    Returns
    -------
    pitches: np.ndarray of bin_indices
    intervals: np.ndarray of rows containing (onset_index, offset_index)
    velocities: np.ndarray of velocity values
    """
    onsets = onsets.numpy()
    frames = frames.numpy()
    velocity = velocity.numpy()

    onsets = onsets > onset_threshold
    frames = frames > frame_threshold

    onsets = onsets.astype(np.uint8)
    frames = frames.astype(np.uint8)

    # Subtract with one-frame offset and look for values of 1. These are the frames at which
    # previous frame was zero and the current frame is one.
    onset_diff = np.concatenate([onsets[:1, :], onsets[1:, :] - onsets[:-1, :]], axis=0) == 1
    onset_diff = onset_diff.astype(np.uint8)  # convert bool to integer

    pitches = []
    intervals = []
    velocities = []

    nonzeros = onset_diff.nonzero()  # nonzero in numpy is a little different than pytorch?
    for frame, pitch in zip(nonzeros[0], nonzeros[1]):

        onset = frame
        offset = frame
        velocity_samples = []

        while onsets[offset, pitch].item() or frames[offset, pitch].item():
            if onsets[offset, pitch].item():
                velocity_samples.append(velocity[offset, pitch].item())
            offset += 1
            if offset == onsets.shape[0]:
                break

        if offset > onset:
            pitches.append(pitch)
            intervals.append([onset, offset])
            velo = np.mean(velocity_samples).item() if len(velocity_samples) > 0 else 0
            velo = max(.0001, velo)  # NB: velocity can't be zero for some mir_eval stuff.
            velocities.append(velo)

    return np.array(pitches), np.array(intervals), np.array(velocities)


def notes_to_frames(pitches, intervals, shape):
    """
    Takes lists specifying notes sequences and return

    Parameters
    ----------
    pitches: list of pitch bin indices
    intervals: list of [onset, offset] ranges of bin indices
    shape: the shape of the original piano roll, [n_frames, n_bins]

    Returns
    -------
    time: np.ndarray containing the frame indices
    freqs: list of np.ndarray, each containing the frequency bin indices
    """
    roll = np.zeros(tuple(shape))
    for pitch, (onset, offset) in zip(pitches, intervals):
        roll[onset:offset, pitch] = 1

    time = np.arange(roll.shape[0])
    freqs = [roll[t, :].nonzero()[0] for t in time]
    return time, freqs
