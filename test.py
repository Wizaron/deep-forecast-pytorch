import torch
import random, os, argparse
import numpy as np
from lib import Data, Model, Loader, draw_graph_all_stations
from settings import Settings

s = Settings()

parser = argparse.ArgumentParser()
parser.add_argument('--data', help='Path of the processed dataset', type=str, required=True)
parser.add_argument('--model', help='Path of the model', type=str, required=True)
parser.add_argument('--n_stations', help='Number of stations to use', type=int, default=5)
parser.add_argument('--usegpu', action='store_true', help='Enable cuda to train on gpu')
args = parser.parse_args()

if torch.cuda.is_available() and not args.usegpu:
    print 'WARNING: You have a CUDA device, so you should probably run with --usegpu'

model_dir = os.path.dirname(args.model)

# Load Seeds
random.seed(s.SEED)
np.random.seed(s.SEED)
torch.manual_seed(s.SEED)

# Load Data
data = Data(data_file=args.data, input_horizon=s.INPUT_HORIZON,
            n_stations=args.n_stations, train_ratio=s.TRAIN_RATIO,
            val_ratio=s.VAL_RATIO, debug=False)

# Load Model
model = Model(args.n_stations, s.MOVING_HORIZON, s.ACTIVATION, s.CRITERION, load_model_path=args.model, usegpu=args.usegpu)

# Train First RNN
_, _, [X_test, y_test] = data.load_data_lstm_1()

print '\n\n' + '#' * 10 + ' TESTING ' + '#' * 10
prediction_test = model.test([X_test, y_test])
draw_graph_all_stations(model_dir, data, args.n_stations, y_test, prediction_test)
