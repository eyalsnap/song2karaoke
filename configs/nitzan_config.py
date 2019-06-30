net_parameters = {
    'data_parameter': {
        'batch_size': 1,
        'num_of_workers': 4
    },
    'model_parametr': {
        'lr': 5e-4,
        'lr_decay': 0.1,
        'lr_decay_epoch_frequency': 3,
        'device': 'cpu',
        'momentum': 0.9
    },
    'weights_saving_parameter': {
        'path': r'C:\dev\free_time_projects\kareoke\output',
        'save_frequency': 2,
        'num_of_files_to_save': 3,
        'save_name': 'MockNet',
        'save_name_prefix': 'weights',
        'should_load': True
    }
}
