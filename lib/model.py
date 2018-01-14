import os, time
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
from itertools import ifilter

from arch import MultiLSTM

class Model(object):

    def __init__(self, in_out_vec_dim, moving_horizon, activation, criterion, load_model_path='', usegpu=True):

        self.in_out_vec_dim = in_out_vec_dim
        self.moving_horizon = moving_horizon
        self.activation = activation
        self.load_model_path = load_model_path
        self.usegpu = usegpu

        self.model = MultiLSTM(self.in_out_vec_dim, self.moving_horizon, self.activation, usegpu=self.usegpu)
        self.__load_weights()
        self.__define_criterion(criterion)

        if self.usegpu:
            cudnn.benchmark = True
            self.model.cuda()
            #self.model = torch.nn.DataParallel(self.model, device_ids=range(self.ngpus))

        print self.model

    def __load_weights(self):

        #def weights_initializer(m):
        #    """Custom weights initialization"""
        #    classname = m.__class__.__name__
        #    if classname.find('Linear') != -1:
        #        m.weight.data.normal_(0.0, 0.001)
        #        m.bias.data.zero_()

        if self.load_model_path != '':
            assert os.path.isfile(self.load_model_path), 'Model : {} does not exists!'.format(self.load_model_path)
            print 'Loading model from {}'.format(self.load_model_path)

            if self.usegpu:
                self.model.load_state_dict(torch.load(self.load_model_path))
            else:
                self.model.load_state_dict(torch.load(self.load_model_path, map_location=lambda storage, loc: storage))

        #else:
        #    self.model.apply(weights_initializer)

    def __define_variable(self, tensor, volatile=False):
        return Variable(tensor, volatile=volatile)

    def __define_input_variables(self, features, targets, volatile=False):
        features_var = self.__define_variable(features, volatile=volatile)
        targets_var = self.__define_variable(targets, volatile=volatile)

        return features_var, targets_var

    def __define_criterion(self, criterion):

       assert criterion in ['MSE', 'L1Loss', 'SmoothL1Loss']

       if criterion == 'MSE':
           self.criterion = torch.nn.MSELoss()
       elif criterion == 'L1Loss':
           self.criterion = torch.nn.L1Loss()
       elif criterion == 'SmoothL1Loss':
           self.criterion = torch.nn.SmoothL1Loss()

    def __define_optimizer(self, learning_rate, weight_decay, lr_drop_factor, lr_drop_patience, rnn_model_num, optimizer='Adam'):
        assert optimizer in ['RMSprop', 'Adam', 'Adadelta', 'SGD']

        for rnn_model_num_counter in range(1, self.moving_horizon + 1):
            if rnn_model_num_counter == rnn_model_num:
                for param in self.model.rnn_models[rnn_model_num_counter - 1].parameters():
                    param.requires_grad = True
            else:
                for param in self.model.rnn_models[rnn_model_num_counter - 1].parameters():
                    param.requires_grad = False

        parameters = ifilter(lambda p: p.requires_grad, self.model.rnn_models[rnn_model_num - 1].parameters())

        if optimizer == 'RMSprop':
            self.optimizer = optim.RMSprop(parameters, lr=learning_rate, weight_decay=weight_decay)
        elif optimizer == 'Adadelta':
            self.optimizer = optim.Adadelta(parameters, lr=learning_rate, weight_decay=weight_decay)
        elif optimizer == 'Adam':
            self.optimizer = optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
        elif optimizer == 'SGD':
            self.optimizer = optim.SGD(parameters, lr=learning_rate, momentum=0.9, weight_decay=weight_decay)

        self.lr_scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=lr_drop_factor, patience=lr_drop_patience, verbose=True)

    @staticmethod
    def __get_loss_averager():
        return averager()

    def __minibatch(self, train_test_iter, rnn_model_num, clip_grad_norm, mode='training'):
        assert mode in ['training', 'test'], 'Mode must be either "training" or "test"'

        if mode == 'training':
            for rnn_model_num_counter in range(1, self.moving_horizon + 1):
                if rnn_model_num_counter == rnn_model_num:
                    for param in self.model.rnn_models[rnn_model_num_counter - 1].parameters():
                        param.requires_grad = True
                else:
                    for param in self.model.rnn_models[rnn_model_num_counter - 1].parameters():
                        param.requires_grad = False
            self.model.train()
        else:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()

        cpu_features, cpu_targets = train_test_iter.next()
        cpu_features = cpu_features.contiguous()
        cpu_targets = cpu_targets.contiguous()

        if self.usegpu:
            gpu_features = cpu_features.cuda(async=True)
            gpu_targets = cpu_targets.cuda(async=True)
        else:
            gpu_features = cpu_features
            gpu_targets = cpu_targets

        if mode == 'training':
            gpu_features, gpu_targets = self.__define_input_variables(gpu_features, gpu_targets)
        else:
            gpu_features, gpu_targets = self.__define_input_variables(gpu_features, gpu_targets, volatile=True)

        predictions = self.model((gpu_features, rnn_model_num))

        cost = self.criterion(predictions, gpu_targets)

        if mode == 'training':
            self.model.zero_grad()
            cost.backward()
            if clip_grad_norm != 0:
                torch.nn.utils.clip_grad_norm(self.model.parameters(), clip_grad_norm)
            self.optimizer.step()

        return cost, predictions, cpu_targets

    def __test(self, rnn_model_num, test_loader):

        print '***** Testing *****'

        n_minibatches = len(test_loader)
        test_iter = iter(test_loader)

        test_loss_averager = Model.__get_loss_averager()

        for minibatch_index in range(n_minibatches):
            loss, predictions, cpu_targets = self.__minibatch(test_iter, rnn_model_num, 0.0, mode='test')
            test_loss_averager.add(loss)

        test_loss = test_loss_averager.val()

        print 'Loss : {}'.format(test_loss)

        return test_loss

    def fit(self, rnn_model_num, learning_rate, weight_decay, clip_grad_norm, lr_drop_factor, lr_drop_patience, patience, optimizer,
            n_epochs, train_loader, test_loader, model_save_path):

        training_log_file = open(os.path.join(model_save_path, 'training.log'), 'w')
        validation_log_file = open(os.path.join(model_save_path, 'validation.log'), 'w')

        training_log_file.write('Epoch,Loss\n')
        validation_log_file.write('Epoch,Loss\n')

        train_loss_averager = Model.__get_loss_averager()

        self.__define_optimizer(learning_rate, weight_decay, lr_drop_factor, lr_drop_patience, rnn_model_num, optimizer=optimizer)

        self.__test(rnn_model_num, test_loader)

        best_val_loss = np.Inf
        n_epochs_wo_best_model = 0
        for epoch in range(n_epochs):
            epoch_start = time.time()

            train_iter = iter(train_loader)
            n_minibatches = len(train_loader)

            minibatch_index = 0
            while minibatch_index < n_minibatches:
                minibatch_loss, minibatch_predictions, minibatch_cpu_targets = self.__minibatch(train_iter, rnn_model_num, clip_grad_norm,
                                                                                                mode='training')

                train_loss_averager.add(minibatch_loss)
                minibatch_index += 1

            train_loss = train_loss_averager.val()

            epoch_end = time.time()
            epoch_duration = epoch_end - epoch_start

            print '[{}] [{}/{}] Loss : {}'.format(epoch_duration, epoch, n_epochs, train_loss)

            val_loss = self.__test(rnn_model_num, test_loader)

            self.lr_scheduler.step(val_loss)

            is_best_model = val_loss <= best_val_loss

            if is_best_model:
                best_val_loss = val_loss
                n_epochs_wo_best_model = 0
                torch.save(self.model.state_dict(), os.path.join(model_save_path, 'model_{}_{}.pth'.format(epoch, val_loss)))
            else:
                n_epochs_wo_best_model += 1

            training_log_file.write('{},{}\n'.format(epoch, train_loss))
            validation_log_file.write('{},{}\n'.format(epoch, val_loss))
            training_log_file.flush()
            validation_log_file.flush()

            train_loss_averager.reset()

            if n_epochs_wo_best_model == patience:
                break

        training_log_file.close()
        validation_log_file.close()

    def test(self, (X_test, y_test)):

        predicted = np.zeros_like(y_test)

        for ind in range(len(X_test)):
            model_ind = ind % self.moving_horizon
            rnn_model_num = model_ind + 1
            if model_ind == 0:
                test_input_raw = X_test[ind]
                #test_input_shape = test_input_raw.shape
                #test_input = np.reshape(test_input_raw, [1, test_input_shape[0], test_input_shape[1]])
                test_input = np.expand_dims(test_input_raw, axis=0)
            else:
                test_input_raw = np.vstack((test_input_raw, predicted[ind - 1]))
                test_input_raw = np.delete(test_input_raw, 0, axis=0)
                #test_input_shape = test_input_raw.shape
                #test_input = np.reshape(test_input_raw, [1, test_input_shape[0], test_input_shape[1]])
                test_input = np.expand_dims(test_input_raw, axis=0)

            predicted[ind] = self.predict(rnn_model_num, test_input)[0]

        return predicted

    def predict(self, rnn_model_num, features):

        assert len(features.shape) == 3 #b, t, feats

        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

        features = torch.from_numpy(features.astype(np.float32))

        features = features.contiguous()
        if self.usegpu:
            features = features.cuda(async=True)

        features = self.__define_variable(features, volatile=True)

        predictions = self.model((features, rnn_model_num))
        predictions = predictions.data.cpu().numpy().astype(np.float32)

        return predictions

class averager(object):
    """Compute average for `torch.Variable` and `torch.Tensor`."""

    def __init__(self):
        self.reset()

    def add(self, v):
        if isinstance(v, Variable):
            count = v.data.numel()
            v = v.data.sum()
        elif isinstance(v, torch.Tensor):
            count = v.numel()
            v = v.sum()

        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res
