from ignite.engine import Events
from ignite.contrib.handlers import LRScheduler
from ignite.handlers import ModelCheckpoint
from torch.optim.lr_scheduler import StepLR


def adding_training_functions(trainer):

    @trainer.on(Events.ITERATION_STARTED)
    def start_of_batch(engine):
        print(f'start of batch {engine.state.iteration}')

    @trainer.on(Events.ITERATION_COMPLETED)
    def end_of_batch(engine):
        print(f'end of batch {engine.state.iteration} - loss {engine.state.output}')

    @trainer.on(Events.EPOCH_STARTED)
    def start_of_epoch(engine):
        print(f'start of epoch {engine.state.epoch}')

    @trainer.on(Events.EPOCH_COMPLETED)
    def end_of_epoch(engine):
        print(f'end of epoch {engine.state.epoch}')


def adding_lr_decay_handler(optimizer, trainer, model_parameters):
    lr_decay_epoch_frequency = model_parameters['lr_decay_epoch_frequency']
    lr_decay = model_parameters['lr_decay']
    step_scheduler = StepLR(optimizer, step_size=lr_decay_epoch_frequency, gamma=lr_decay)
    scheduler = LRScheduler(step_scheduler)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, scheduler)


def adding_weight_save_handler(model, trainer, weight_save_config_parameters):
    weight_dir = weight_save_config_parameters['path']
    saving_frequecy = weight_save_config_parameters['save_frequency']
    saving_name = weight_save_config_parameters['save_name']
    saving_prefix = weight_save_config_parameters['save_name_prefix']
    handler = ModelCheckpoint(weight_dir,
                              saving_prefix,
                              save_interval=saving_frequecy,
                              create_dir=True)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, handler, {saving_name: model})
