import os

import torch
from torch.utils.data import DataLoader
from data.audio_dataset import AudioDataset
from ignite.engine import create_supervised_trainer, Events
from torch import optim

from nets.MockLoss import MockLoss
from nets.mock_net import MockNet
from runners.adding_handler_utils import adding_training_functions, adding_weight_save_handler, adding_lr_decay_handler


def build_dataloader(dir_paths):
    audio_dataset = AudioDataset(dir_paths)
    return DataLoader(audio_dataset,
                      batch_size=1,
                      shuffle=False,
                      num_workers=4)


def build_optimizer(net_param, model_config_parameters):
    learning_rate = model_config_parameters['lr']
    momentum = model_config_parameters['momentum']

    return optim.SGD(
        params=net_param,
        lr=learning_rate,
        momentum=momentum
    )


def build_basic_ignite_trainer(net_config_parameters):

    model = build_net(net_config_parameters['weights_saving_parameter'])
    optimizer = build_optimizer(model.parameters(), net_config_parameters['model_parametr'])
    loss = MockLoss()

    trainer = create_supervised_trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss,
        device='cpu'
    )

    adding_weight_save_handler(model, trainer, net_config_parameters['weights_saving_parameter'])

    adding_lr_decay_handler(optimizer, trainer, net_config_parameters['model_parametr'])

    return trainer


def build_net(weight_parameters):

    model = MockNet()
    if not weight_parameters['should_load']:
        return model

    file_name = os.listdir(weight_parameters['path'])[-1]
    weight_path = os.path.join(weight_parameters['path'], file_name)
    model.load_state_dict(torch.load(weight_path))
    print(f'loading weights from file {file_name}')
    return model


def build_trainer(net_parameters):
    trainer = build_basic_ignite_trainer(net_parameters)
    adding_training_functions(trainer)
    return trainer
