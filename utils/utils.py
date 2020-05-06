from __future__ import print_function, division
import torch
import yaml
import os

from models.Conv import conv_model
from models.FC import FC_model
from models.RNN import LSTM_model, LSTM_model2
from loaders.loadNGSIM import NGSIMDataset
# from loadArgoverse import ArgoverseDataset
from loaders.loadFusion import FusionDataset


class Settings:
    class __Settings:
        def __init__(self):
            self.settings_dict = yaml.safe_load(open('utils/settings.yaml'))
            self.refresh()

        def refresh(self):
            if self.settings_dict['device'] == '':
                self.settings_dict['device'] = 'cpu'
                if torch.cuda.is_available():
                    self.settings_dict['device'] = 'cuda'
                    print('Using device ' + torch.cuda.get_device_name())
            self.settings_dict['use_yaw'] = self.settings_dict['model_type'][:7] == 'Bicycle'
            self.settings_dict['name'] = (self.settings_dict['model_type'] + '_' +
                                          self.settings_dict['dataset'] + '_' +
                                          str(self.settings_dict['training_id']))
            self.settings_dict['log_path'] = ('./logs/' )
            self.settings_dict['models_path'] = ('./trained_models/')
            if self.settings_dict['dataset'] == 'NGSIM':
                self.settings_dict['dt'] = 0.1*self.settings_dict['down_sampling']
                self.settings_dict['unit_conversion'] = 0.3048
                self.settings_dict['time_hist'] = 3
                self.settings_dict['time_pred'] = min(5, self.settings_dict['time_pred'])
                self.settings_dict['field_height'] = 30
                self.settings_dict['field_width'] = 120
            elif self.settings_dict['dataset'] == 'Argoverse':
                self.settings_dict['dt'] = 0.1*self.settings_dict['down_sampling']
                self.settings_dict['unit_conversion'] = 1
                self.settings_dict['time_hist'] = 2
                self.settings_dict['time_pred'] = min(3, self.settings_dict['time_pred'])
                self.settings_dict['field_height'] = 120
                self.settings_dict['field_width'] = 120
            elif self.settings_dict['dataset'] == 'Fusion':
                self.settings_dict['dt'] = 0.04*self.settings_dict['down_sampling']
                self.settings_dict['unit_conversion'] = 1
                self.settings_dict['time_hist'] = 2
                self.settings_dict['time_pred'] = min(3, self.settings_dict['time_pred'])
                self.settings_dict['field_height'] = 120
                self.settings_dict['field_width'] = 120
            else:
                raise ValueError('The dataset "' + self.settings_dict['dataset'] + '" is unknown. Please correct the'
                                 'dataset name in "settings.yaml" or modify the Settings class in "utils.py" to handle it.')

        def __str__(self):
            return repr(self) + self.settings_dict

    instance = None
    def __init__(self):
        if not Settings.instance:
            Settings.instance = Settings.__Settings()
        else:
            pass
    def __getattr__(self, name):
        return self.instance.settings_dict[name]

    def __setattr__(self, name, value):
        self.instance.settings_dict[name] = value
        self.instance.refresh()

    def get_dict(self):
        return self.instance.settings_dict.copy()


def get_dataset():
    args = Settings()
    if args.dataset == 'NGSIM':
        trSet = NGSIMDataset( args.NGSIM_data_directory + 'TrainSet_traj_v2.mat',
                              args.NGSIM_data_directory + 'TrainSet_tracks_v2.mat', args=args)
        valSet = NGSIMDataset(args.NGSIM_data_directory + 'ValSet_traj_v2.mat',
                              args.NGSIM_data_directory + 'ValSet_tracks_v2.mat', args=args)
    elif args.dataset == 'Argoverse':
        trSet = ArgoverseDataset(args.argoverse_data_directory + 'train/data', args=args)
        valSet = ArgoverseDataset(args.argoverse_data_directory + 'val/data', args=args)
    elif args.dataset == 'Fusion':
        trSet = FusionDataset(args.fusion_data_directory + 'train_sequenced_data.tar', args=args)
        valSet = FusionDataset(args.fusion_data_directory + 'val_sequenced_data.tar', args=args)

    return trSet, valSet


def get_test_set():
    args = Settings()
    if args.dataset == 'NGSIM':
        testSet = NGSIMDataset(args.NGSIM_test_data_directory + 'TestSet_traj_v2.mat',
                             args.NGSIM_test_data_directory + 'TestSet_tracks_v2.mat', args)
    elif args.dataset == 'Argoverse':
        testSet = ArgoverseDataset(args.argoverse_data_directory + 'val/data', args)
    elif args.dataset == 'Fusion':
        testSet = FusionDataset(args.fusion_data_directory + 'test_sequenced_data.tar', args)

    return testSet


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_net():
    args = Settings()
    if args.model_type == 'FC':
        net = FC_model(args)
    elif args.model_type == 'conv':
        net = conv_model(args)
    elif args.model_type == 'LSTM':
        net = LSTM_model(args)
    elif args.model_type == 'LSTM2':
        net = LSTM_model2(args)
    else:
        print('Model type ' + args.model_type + ' is not known.')

    net = net.to(args.device)
    print("Net number of parameters: %d" % count_parameters(net))

    if args.load_name != '':
        try:
            print('Loaded ' + args.load_name)
            net.load_state_dict(torch.load('./trained_models/unique_object/' + args.model_type + '/' + args.load_name + '.tar', map_location=args.device))
        except RuntimeError as err:
            print(err)
            print('Loading what can be loaded with option strict=False.')
            net.load_state_dict(torch.load('./trained_models/unique_object/' + args.model_type + '/' + args.load_name + '.tar', map_location=args.device), strict=False)
    return net

def make_dir(path):
    try:
        os.makedirs(path)
    except OSError:
        pass


