import numpy as np
import csv
np.random.seed(1234)

class Data(object):

    def __init__(self, data_file='data/data_imputation_turkey.csv', input_horizon=12, n_stations=27,
                 train_ratio=0.985, val_ratio=0.05, debug=False):
        """
        Inputs
        ======
        :param data_file: input csv filepath.
        :param input_horizon:
        :param n_stations:
        :param train_ratio:
        :param debug:"""

        self.data_file = data_file
        self.input_horizon = input_horizon
        self.n_stations = n_stations
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio

        self.trainDataRate = 1 if not debug else 0.05 # percentage of data used for training (saving time for debuging)

        self.winds = Data.read_data(self.data_file, self.n_stations)
        self.winds, self.means_stds = Data.normalize_data(self.winds)

    @staticmethod
    def read_data(data_file, n_stations):
        with open(data_file) as f:
            data = csv.reader(f, delimiter=',')

            winds = [line for line in data]
            winds = np.array(winds).astype(np.float32)
            winds = winds[:, : n_stations]

        return winds

    @staticmethod
    def normalize_data(winds):
        wind_min = winds.min()
        wind_max = winds.max() - wind_min

        normal_winds = (winds - wind_min) / wind_max
        mins_maxs = [wind_min, wind_max]

        return normal_winds, mins_maxs

    def denormalize_data(self, vec):
        wind_min, wind_max = self.means_stds
        res = vec * wind_max + wind_min
        return res

    def load_data_lstm_1(self): # For LSTM 1

        samples = []
        for index in range(self.winds.shape[0] - self.input_horizon): # Last one is reserved for label
            samples.append(self.winds[index : index + self.input_horizon])
        samples = np.array(samples)

        n_samples = samples.shape[0]
        n_train_samples = int(round(n_samples * self.train_ratio))
        n_train_samples = int(round(n_train_samples * self.trainDataRate))

        X_train = samples[: n_train_samples, :]
        y_train = self.winds[self.input_horizon : n_train_samples + self.input_horizon] # Shifted by self.input_horizon

        X_test = samples[n_train_samples : n_samples]
        y_test = self.winds[n_train_samples + self.input_horizon : n_samples + self.input_horizon]

        n_val_samples = int(np.ceil(n_train_samples * self.val_ratio))
        X_val = X_train[: n_val_samples]
        y_val = y_train[: n_val_samples]

        X_train = X_train[n_val_samples :]
        y_train = y_train[n_val_samples :]

        return [X_train, y_train], [X_val, y_val], [X_test, y_test]

    def load_data(self, pre_x_train_val, pre_y_train_val, model, rnn_model_num): # pre_x_train_val and pre_y_train_val from load_data_lstm_1

        X_train_val, y_train_val = np.ones_like(pre_x_train_val), np.zeros_like(pre_y_train_val)

        for ind in range(len(pre_x_train_val) - 1):
            tempInput = pre_x_train_val[ind]
            temp_shape = tempInput.shape
            tempInput = np.reshape(tempInput, (1, temp_shape[0], temp_shape[1]))

            output = model.predict(rnn_model_num, tempInput)

            tInput = np.reshape(tempInput, temp_shape)
            tempInput = np.vstack((tInput, output))
            tempInput = np.delete(tempInput, 0, axis=0)

            X_train_val[ind] = tempInput
            y_train_val[ind] = pre_y_train_val[ind + 1]

        X_train_val = X_train_val[:-1]
        y_train_val = y_train_val[:-1]

        return [X_train_val, y_train_val]
