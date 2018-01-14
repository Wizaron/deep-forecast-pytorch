import torch.nn as nn

class MultiLSTM(nn.Module):

    def __init__(self, in_out_vec_dim, moving_horizon, activation, usegpu=True):
        super(MultiLSTM, self).__init__()

        assert activation in ['relu', 'sigmoid', 'elu']

        self.n_inputs = in_out_vec_dim
        self.n_outputs = in_out_vec_dim
        self.moving_horizon = moving_horizon
        self.activation = activation
        self.usegpu = usegpu

        self.rnn_models = nn.ModuleList()
        for model_idx in range(1, self.moving_horizon + 1):
            self.rnn_models.append(nn.ModuleList(self.build_rnn(model_idx)))

    def build_single_layer_rnn(self, n_units):

        rnn = nn.LSTM(self.n_inputs, n_units, batch_first=True)   #TODO: Bidi + attention
        output = nn.Linear(n_units, self.n_outputs)
        if self.activation == 'relu':
            act = nn.ReLU()
        elif self.activation == 'sigmoid':
            act = nn.Sigmoid()
        elif self.activation == 'elu':
            act = nn.ELU()

        return [rnn, output, act]

    def build_two_layer_rnn(self, n_units_1, n_units_2):

        rnn_1 = nn.LSTM(self.n_inputs, n_units_1, batch_first=True)
        rnn_2 = nn.LSTM(n_units_1, n_units_2, batch_first=True)   #TODO: Bidi + attention
        output = nn.Linear(n_units_2, self.n_outputs)
        if self.activation == 'relu':
            act = nn.ReLU()
        elif self.activation == 'sigmoid':
            act = nn.Sigmoid()
        elif self.activation == 'elu':
            act = nn.ELU()

        return [rnn_1, rnn_2, output, act]

    def build_rnn_1(self):

        return self.build_single_layer_rnn(self.n_inputs * 2)

    def build_rnn_2(self):

        #return self.build_single_layer_rnn(10)
        return self.build_single_layer_rnn(self.n_inputs * 2)

    def build_rnn_3(self):

        #return self.build_single_layer_rnn(self.n_inputs)
        return self.build_single_layer_rnn(self.n_inputs * 2)

    def build_rnn_4(self):

        #return self.build_two_layer_rnn(self.n_inputs, self.n_inputs * 2)
        return self.build_single_layer_rnn(self.n_inputs * 2)

    def build_rnn_5(self):

        #return self.build_single_layer_rnn(30)
        return self.build_single_layer_rnn(self.n_inputs * 2)

    def build_rnn_6(self):

        #return self.build_two_layer_rnn(self.n_inputs * 2, self.n_inputs * 2)
        return self.build_single_layer_rnn(self.n_inputs * 2)

    def build_rnn(self, rnn_model_num):

        if rnn_model_num == 1:
            return self.build_rnn_1()
        elif rnn_model_num == 2:
            return self.build_rnn_2()
        elif rnn_model_num == 3:
            return self.build_rnn_3()
        elif rnn_model_num == 4:
            return self.build_rnn_4()
        elif rnn_model_num == 5:
            return self.build_rnn_5()
        elif rnn_model_num == 6:
            return self.build_rnn_6()
        else:
            return self.build_rnn_6()

    def forward(self, (x, rnn_model_num)):

        rnn_model = self.rnn_models[rnn_model_num - 1]

        if len(rnn_model) == 3:
            rnn_layer, output_layer, activation = rnn_model
            _, x = rnn_layer(x)
            x = x[0]
            x = x.squeeze(0)
            x = output_layer(x)
            x = activation(x)

        elif len(rnn_model) == 4:
            rnn_layer_1, rnn_layer_2, output_layer, activation = rnn_model
            x, _ = rnn_layer_1(x)
            _, x = rnn_layer_2(x)
            x = x[0]
            x = x.squeeze(0)
            x = output_layer(x)
            x = activation(x)
        else:
            NotImplementedError

        return x
