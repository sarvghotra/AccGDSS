import torch
import argparse
import time
from parsers.parser import Parser
from parsers.config import get_config
from trainer import Trainer
from sampler import Sampler, Sampler_mol


def main(work_type_args):
    ts = time.strftime('%b%d-%H:%M:%S', time.gmtime())
    args = Parser().parse()
    config = get_config(args.config, args.seed)

    # -------- Train --------
    if work_type_args.type == 'train':
        raise ValueError(f'Wrong type : {work_type_args.type}')
        trainer = Trainer(config)
        ckpt = trainer.train(ts)
        if 'sample' in config.keys():
            config.ckpt = ckpt
            sampler = Sampler(config)
            sampler.sample()

    # -------- Generation --------
    elif work_type_args.type == 'sample':
        if config.data.data in ['QM9', 'ZINC250k']:
            sampler = Sampler_mol(config)
        else:
            sampler = Sampler(config)

        if config.sampler.dpm_solver:
            # FIXME: set n_samples to 1000(0)
            assert config.sampler.dpm_config.steps is not None
            sampler.dpm_solver_sampling(n_samples=64, diff_steps=config.sampler.dpm_config.steps)
        else:
            # FIXME: set n_samples to 1000(0)
            sampler.sample(n_samples=64)

    else:
        raise ValueError(f'Wrong type : {work_type_args.type}')

if __name__ == '__main__':

    work_type_parser = argparse.ArgumentParser()
    work_type_parser.add_argument('--type', type=str, required=True)
    main(work_type_parser.parse_known_args()[0])
