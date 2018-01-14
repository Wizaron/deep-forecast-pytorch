import torch, os
from lib import Data, Model, Loader, draw_graph_all_stations

debug = False
n_workers = 4
batch_size = 256
activation = 'relu' #sigmoid, relu, elu
input_horizon = 12
n_stations = 5
train_ratio = 0.985
learning_rate = 0.01 #0.001
moving_horizon = 6
n_epochs = [100, 100, 100, 100, 100, 100] if not debug else [1, 1, 1, 1, 1, 1]
val_ratio = 0.05
criterion = 'L1Loss' #L1Loss, MSE, SmoothL1Loss
usegpu = False
pin_memory = False
if usegpu:
    pin_memory = True

model_save_path = './models/models_{}'
output_dir = './outputs'

assert len(n_epochs) == moving_horizon

for rnn_model_num in range(1, moving_horizon + 1):
    try:
        os.makedirs(model_save_path.format(rnn_model_num))
    except:
        pass

try:
    os.makedirs(output_dir)
except:
    pass

data = Data(data_file='data/processed/turkey_2016/data_imputation_turkey.csv', input_horizon=input_horizon,
            n_stations=n_stations, train_ratio=train_ratio,
            val_ratio=val_ratio, debug=debug)
model = Model(n_stations, moving_horizon, activation, criterion, usegpu=usegpu)

weight_decay = 1e-4
clip_grad_norm = 40
lr_drop_factor = 0.1
lr_drop_patience = 10
patience = 20
optimizer = 'RMSprop'

[X_train, y_train], [X_val, y_val], [X_test, y_test] = data.load_data_lstm_1()

rnn_model_num = 1
print '######### RNN 1 ##############'

train_loader = torch.utils.data.DataLoader(Loader((X_train, y_train)), batch_size=batch_size, shuffle=True,
                                           num_workers=n_workers, pin_memory=pin_memory)

val_loader = torch.utils.data.DataLoader(Loader((X_val, y_val)), batch_size=batch_size, shuffle=False,
                                         num_workers=n_workers, pin_memory=pin_memory)

model.fit(rnn_model_num, learning_rate, weight_decay, clip_grad_norm, lr_drop_factor, lr_drop_patience, patience, 
          optimizer, n_epochs[rnn_model_num - 1],
          train_loader, val_loader, model_save_path.format(rnn_model_num))

for rnn_model_num in range(2, moving_horizon + 1):
    X_train, y_train = data.load_data(X_train, y_train, model, rnn_model_num - 1)
    X_val, y_val = data.load_data(X_val, y_val, model, rnn_model_num - 1)
    print '######### RNN {} ##############'.format(rnn_model_num)
    train_loader = torch.utils.data.DataLoader(Loader((X_train, y_train)), batch_size=batch_size, shuffle=True,
                                               num_workers=n_workers, pin_memory=pin_memory)

    val_loader = torch.utils.data.DataLoader(Loader((X_val, y_val)), batch_size=batch_size, shuffle=False,
                                             num_workers=n_workers, pin_memory=pin_memory)

    model.fit(rnn_model_num, learning_rate, weight_decay, clip_grad_norm, lr_drop_factor, lr_drop_patience, patience,
              optimizer, n_epochs[rnn_model_num - 1],
              train_loader, val_loader, model_save_path.format(rnn_model_num))

print 'TESTING ...'
prediction_test = model.test([X_test, y_test])

draw_graph_all_stations(output_dir, data, n_stations, y_test, prediction_test)
