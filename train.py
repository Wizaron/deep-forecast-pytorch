import torch
import random, getpass, datetime, shutil, os, argparse
import numpy as np
from lib import Data, Model, Loader, draw_graph_all_stations
from settings import Settings

s = Settings()

parser = argparse.ArgumentParser()
parser.add_argument('--data', help='Path of the processed dataset', type=str, required=True)
parser.add_argument('--n_stations', help='Number of stations to use', type=int, default=5)
parser.add_argument('--batch_size', help='Input minibatch size', type=int, default=256)
parser.add_argument('--n_workers', help='Number of data loading workers [0 to do it using main process]', type=int, default=4)
parser.add_argument('--usegpu', action='store_true', help='Enable cuda to train on gpu')
parser.add_argument('--debug', action='store_true', help='Enable debug mode')
args = parser.parse_args()

if torch.cuda.is_available() and not args.usegpu:
    print 'WARNING: You have a CUDA device, so you should probably run with --usegpu'

def generate_run_id():

    username = getpass.getuser()

    now = datetime.datetime.now()
    date = map(str, [now.year, now.month, now.day])
    coarse_time = map(str, [now.hour, now.minute])
    fine_time = map(str, [now.second, now.microsecond])

    run_id = '_'.join(['-'.join(date), '-'.join(coarse_time), username, '-'.join(fine_time)])
    return run_id

RUN_ID = generate_run_id()
model_save_path = os.path.join('models', RUN_ID, 'models_{}')
output_dir = os.path.join('outputs', RUN_ID)

for rnn_model_num in range(1, s.MOVING_HORIZON + 1):
    try:
        os.makedirs(model_save_path.format(rnn_model_num))
    except:
        pass

try:
    os.makedirs(output_dir)
except:
    pass

pin_memory = False
if args.usegpu:
    pin_memory = True

# Load Seeds
random.seed(s.SEED)
np.random.seed(s.SEED)
torch.manual_seed(s.SEED)

# Load Data
data = Data(data_file=args.data, input_horizon=s.INPUT_HORIZON,
            n_stations=args.n_stations, train_ratio=s.TRAIN_RATIO,
            val_ratio=s.VAL_RATIO, debug=args.debug)

# Load Model
model = Model(args.n_stations, s.MOVING_HORIZON, s.ACTIVATION, s.CRITERION, usegpu=args.usegpu)

# Train First RNN
[X_train, y_train], [X_val, y_val], [X_test, y_test] = data.load_data_lstm_1()

rnn_model_num = 1
print '#' * 10 + ' RNN 1 ' + '#' * 10

train_loader = torch.utils.data.DataLoader(Loader((X_train, y_train)), batch_size=args.batch_size, shuffle=True,
                                           num_workers=args.n_workers, pin_memory=pin_memory)

val_loader = torch.utils.data.DataLoader(Loader((X_val, y_val)), batch_size=args.batch_size, shuffle=False,
                                         num_workers=args.n_workers, pin_memory=pin_memory)

model.fit(rnn_model_num, s.LEARNING_RATE, s.WEIGHT_DECAY, s.CLIP_GRAD_NORM, s.LR_DROP_FACTOR, s.LR_DROP_PATIENCE, s.PATIENCE, 
          s.OPTIMIZER, s.N_EPOCHS[rnn_model_num - 1],
          train_loader, val_loader, model_save_path.format(rnn_model_num))

# Train Other RNNs
for rnn_model_num in range(2, s.MOVING_HORIZON + 1):
    X_train, y_train = data.load_data(X_train, y_train, model, rnn_model_num - 1)
    X_val, y_val = data.load_data(X_val, y_val, model, rnn_model_num - 1)
    print '#' * 10 + ' RNN {} '.format(rnn_model_num) + '#' * 10
    train_loader = torch.utils.data.DataLoader(Loader((X_train, y_train)), batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.n_workers, pin_memory=pin_memory)

    val_loader = torch.utils.data.DataLoader(Loader((X_val, y_val)), batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.n_workers, pin_memory=pin_memory)

    model.fit(rnn_model_num, s.LEARNING_RATE, s.WEIGHT_DECAY, s.CLIP_GRAD_NORM, s.LR_DROP_FACTOR, s.LR_DROP_PATIENCE, s.PATIENCE,
              s.OPTIMIZER, s.N_EPOCHS[rnn_model_num - 1],
              train_loader, val_loader, model_save_path.format(rnn_model_num))


print '\n\n' + '#' * 10 + ' TESTING ' + '#' * 10
prediction_test = model.test([X_test, y_test])
draw_graph_all_stations(output_dir, data, args.n_stations, y_test, prediction_test)
