from tensorflow import keras

class BidirectionalLSTM(keras.Sequential):

    def __init__(self, input_features, recurrent_features):

        super(BidirectionalLSTM, self).__init__([
            keras.layers.InputLayer(input_shape=(None, input_features)),
            keras.layers.Bidirectional(
                keras.layers.LSTM(recurrent_features, recurrent_activation='sigmoid', implementation=2,
                                  return_sequences=True))
        ])
