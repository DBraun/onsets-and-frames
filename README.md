# TensorFlow 2.1 Implementation of Onsets and Frames

This is a [TensorFlow 2.1](https://www.tensorflow.org/) implementation of Google's [Onsets and Frames](https://magenta.tensorflow.org/onsets-frames) model, using the [MAESTRO dataset](https://magenta.tensorflow.org/datasets/maestro) for training and the Disklavier portion of the [MAPS database](http://www.tsi.telecom-paristech.fr/aao/en/2010/07/08/maps-database-a-piano-database-for-multipitch-estimation-and-automatic-transcription-of-music/) for testing. The codebase was ported from Jong Wook Kim's [PyTorch implementation](https://github.com/jongwook/onsets-and-frames), which was based on Google Magenta's TensorFlow 1.x implementation.

## Instructions

This project is quite resource-intensive; 32 GB or larger system memory and 8 GB or larger GPU memory are recommended. During training, the batch size will automatically be halved if an 8 GB card is detected.

### Downloading the Datasets

The `data` subdirectory already contains the MAPS database. To download the MAESTRO dataset, first make sure that you have `ffmpeg` executable and run `prepare_maestro.sh` script:

```
ffmpeg -version
cd data
./prepare_maestro.sh
```

This will download the full MAESTRO dataset (103 GB) from Google's server and automatically unzip and encode the WAV files as FLAC files. You'll still need over 200 GB of space for this step. In the PyTorch project, FLAC files get turned into `.pt` files for faster processing. In this TensorFlow project, the equivalent step uses [TensorLayer](https://tensorlayer.readthedocs.io/en/latest/) to save a dictionary of Tensors to `.npy` files. This increases the necessary intermediate storage by a significant factor, but you can remove FLAC files once you have the `.npy` files.

### Training

All package requirements are contained in `requirements.txt`. To train the model, run:

```
pip install -r requirements.txt
python train.py
```

Trained models will be saved using checkpoints in `runs/run_%Y_%m_%d-%H_%M_%S`.

You can watch the training with TensorBoard:

`tensorboard --logdir=./runs --port=6006`

### Evaluation and Transcription

To evaluate the trained model using the MAPS database, run the following command (substituting your `run` folder) to calculate the note and frame metrics:

```
python evaluate.py --checkpoint-dir runs/run_2020_03_25-15_33_13 --save-path evaluated
```

Specifying a directory for `--save-path` will output the transcribed MIDI files along with the piano roll images.

To transcribe FLAC files to MIDI and piano roll images:

```
python transcribe.py --checkpoint-dir runs/run_2020_03_25-15_33_13 --flac-paths a_flac_folder/*.flac --save-path a_flac_folder
```

That will place the MIDI files next to the FLACs.

## Results

Training 100 epochs on MAESTRO and evaluating on MAPS:

                              note precision                : 0.843 ± 0.067
                              note recall                   : 0.648 ± 0.116
                              note f1                       : 0.729 ± 0.093
                              note overlap                  : 0.521 ± 0.076
                 note-with-offsets precision                : 0.352 ± 0.112
                 note-with-offsets recall                   : 0.271 ± 0.096
                 note-with-offsets f1                       : 0.305 ± 0.101
                 note-with-offsets overlap                  : 0.802 ± 0.093
                note-with-velocity precision                : 0.653 ± 0.088
                note-with-velocity recall                   : 0.505 ± 0.116
                note-with-velocity f1                       : 0.567 ± 0.104
                note-with-velocity overlap                  : 0.523 ± 0.077
    note-with-offsets-and-velocity precision                : 0.280 ± 0.094
    note-with-offsets-and-velocity recall                   : 0.217 ± 0.083
    note-with-offsets-and-velocity f1                       : 0.243 ± 0.087
    note-with-offsets-and-velocity overlap                  : 0.800 ± 0.093
                             frame f1                       : 0.563 ± 0.087
                             frame precision                : 0.701 ± 0.163
                             frame recall                   : 0.493 ± 0.097
                             frame accuracy                 : 0.397 ± 0.084
                             frame substitution_error       : 0.108 ± 0.076
                             frame miss_error               : 0.399 ± 0.129
                             frame false_alarm_error        : 0.166 ± 0.214
                             frame total_error              : 0.673 ± 0.199
                             frame chroma_precision         : 0.751 ± 0.151
                             frame chroma_recall            : 0.533 ± 0.104
                             frame chroma_accuracy          : 0.438 ± 0.079
                             frame chroma_substitution_error: 0.068 ± 0.050
                             frame chroma_miss_error        : 0.399 ± 0.129
                             frame chroma_false_alarm_error : 0.166 ± 0.214
                             frame chroma_total_error       : 0.633 ± 0.179

## Thanks

* [Jong Wook Kim](https://github.com/jongwook/)
* [Magenta](https://magenta.tensorflow.org/)
 

 ## TODO:
 When processing the MAPS MIDI files for training and evaluation, we first translate “sustain pedal” control changes into longer note durations. If a note is active when sustain goes on, that note will be extended until either sustain goes off or the same note is played again. This process gives the same note durations as the text files included with the dataset.
