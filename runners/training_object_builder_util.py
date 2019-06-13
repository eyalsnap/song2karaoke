from ignite.handlers import ModelCheckpoint
from torch.utils.data import DataLoader
from data.audio_dataset import AudioDataset
from ignite.engine import create_supervised_trainer, Events
from torch import optim

from nets.MockLoss import MockLoss
from nets.mock_net import MockNet
from runners.adding_handler_utils import adding_training_functions


def build_dataloader(dir_paths):
    audio_dataset = AudioDataset(dir_paths)
    return DataLoader(audio_dataset,
                      batch_size=1,
                      shuffle=False,
                      num_workers=4)


def build_optimizer(net_param):
    learning_rate = 1e-4
    return optim.SGD(
        params=net_param,
        lr=learning_rate,
        momentum=0.9,
    )


def build_basic_ignite_trainer():
    model = MockNet()
    net_param = model.parameters()
    optimizer = build_optimizer(net_param)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    loss = MockLoss()
    trainer = create_supervised_trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss,
        device='cpu'
    )

    weight_dir = r'C:\Users\Eyal\Desktop\eyal\python\weights\karaoke'
    handler = ModelCheckpoint(weight_dir,
                              'myprefix',
                              save_interval=1000,
                              create_dir=True)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, handler, {'mymodel': model})

    return trainer, optimizer


def build_trainer():
    trainer, optimizer = build_basic_ignite_trainer()
    adding_training_functions(trainer,optimizer)
    return trainer
