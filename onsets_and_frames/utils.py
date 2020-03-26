from PIL import Image
import numpy as np

def save_pianoroll(path, onsets, frames, onset_threshold=0.5, frame_threshold=0.5, zoom=4):
    """
    Saves a piano roll diagram

    Parameters
    ----------
    path: str
    onsets: tensorflow.Tensor, shape = [frames, bins]
    frames: tensorflow.Tensor, shape = [frames, bins]
    onset_threshold: float
    frame_threshold: float
    zoom: int
    """

    onsets = onsets.numpy()
    frames = frames.numpy()

    onsets = (1 - (np.transpose(onsets) > onset_threshold).astype(np.uint8))
    frames = (1 - (np.transpose(frames) > frame_threshold).astype(np.uint8))
    both = (1 - (1 - onsets) * (1 - frames))

    image = np.stack([onsets,frames,both], axis=2)
    image = np.flip(image, axis=0)*255

    image = Image.fromarray(image, 'RGB')
    image = image.resize((image.size[0], image.size[1] * zoom)) # default nearest neighbor resize
    image.save(path)
