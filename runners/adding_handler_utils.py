from ignite.engine import Events


def adding_training_functions(trainer,optimizer):

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
        optimizer.step()
        print(f'end of epoch {engine.state.epoch}')
