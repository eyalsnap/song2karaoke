net_parameters = {
    'data_parameter': {
        'batch_size': 1,
        'num_of_workers': 4
    },
    'model_parametr': {
        'lr': 2e-4,
        'lr_decay': 0.1,
        'lr_decay_epoch_frequency': 3,
        'device': 'cuda',
        'momentum': 0.9
    },
    'weights_saving_parameter': {
        'path': r'output/weights',
        'save_frequency': 2,
        'num_of_files_to_save': 3,
        'save_name': 'MockNet',
        'save_name_prefix': 'weights',
        'should_load': False
    }
}
