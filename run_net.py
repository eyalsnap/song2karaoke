from configs.net_config import net_parameters
from runners.training_object_builder_util import build_dataloader, build_trainer


def main():
    dir_paths = [
        # r'C:\Users\Eyal\Desktop\eyal\python\data\sample_data\sample_data\_nopr_slman__am_ate_shoma_',
        # nitzan comment for testing
        r'C:\Users\Eyal\Desktop\eyal\python\data\sample_data\sample_data\_aiti_loi__zz_btirof_'
    ]# * 10

    audio_dataloader = build_dataloader(dir_paths)
    trainer = build_trainer(net_parameters)
    trainer.run(audio_dataloader, max_epochs=140)


if __name__ == '__main__':
    main()
