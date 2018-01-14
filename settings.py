class Settings(object):

    def __init__(self, debug=False):

        self.INPUT_HORIZON = 12
        self.MOVING_HORIZON = 6

        self.ACTIVATION = 'relu'   # One of 'sigmoid', 'relu', 'elu'
        self.CRITERION = 'L1Loss'  # One of 'L1Loss', 'MSE', 'SmoothL1Loss'
        self.LEARNING_RATE = 0.01
        self.WEIGHT_DECAY = 1e-3
        self.CLIP_GRAD_NORM = 40
        self.LR_DROP_FACTOR = 0.1
        self.LR_DROP_PATIENCE = 10
        self.PATIENCE = 25
        self.OPTIMIZER = 'RMSprop' # One of 'RMSprop', 'Adam', 'Adadelta', 'SGD'
        self.N_EPOCHS = [100, 100, 100, 100, 100, 100] if not debug else [1, 1, 1, 1, 1, 1]

        self.TRAIN_RATIO = 0.98
        self.VAL_RATIO = 0.1

        self.SEED = 73

        assert len(self.N_EPOCHS) == self.MOVING_HORIZON
