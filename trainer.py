import os
import time
from tqdm import tqdm, trange
import numpy as np
import torch
import wandb

from utils.loader import load_seed, load_device, load_data, load_model_params, load_model_optimizer, \
                         load_ema, load_loss_fn, load_batch
from utils.logger import Logger, set_log, start_log, train_log


class Trainer(object):
    def __init__(self, config):
        super(Trainer, self).__init__()

        self.config = config
        self.log_folder_name, self.log_dir, self.ckpt_dir = set_log(self.config)

        self.seed = load_seed(self.config.seed)
        self.device = load_device()
        self.train_loader, self.test_loader = load_data(self.config)

        self.params_x, self.params_adj = load_model_params(self.config)

        # if config.data.data in ['QM9', 'ZINC250k']:
        #     self.sampler =

        wandb.init(project="GGM_proj_molecule_gen",
                    entity="sarvghotra",
                    notes="Logging testing",
                    tags=["debug", "gdss"],
                    name=self.config.train.name,
                    config=self.config,
                    allow_val_change=True,
                    resume="allow",)


    def train(self, ts):
        self.config.exp_name = ts
        self.ckpt = f'{ts}'
        print('\033[91m' + f'{self.ckpt}' + '\033[0m')

        # -------- Load models, optimizers, ema --------
        self.model_x, self.optimizer_x, self.scheduler_x = load_model_optimizer(self.params_x, self.config.train,
                                                                                self.device)
        self.model_adj, self.optimizer_adj, self.scheduler_adj = load_model_optimizer(self.params_adj, self.config.train,
                                                                                        self.device)
        self.ema_x = load_ema(self.model_x, decay=self.config.train.ema)
        self.ema_adj = load_ema(self.model_adj, decay=self.config.train.ema)

        logger = Logger(str(os.path.join(self.log_dir, f'{self.ckpt}.log')), mode='a')
        logger.log(f'{self.ckpt}', verbose=False)
        wandb.config.update({
            "ckpt": self.ckpt,
        })
        start_log(logger, self.config)
        train_log(logger, self.config)

        self.loss_fn = load_loss_fn(self.config)

        # -------- Training --------
        for epoch in trange(0, (self.config.train.num_epochs), desc = '[Epoch]', position = 1, leave=False):

            self.train_x = []
            self.train_adj = []
            self.test_x = []
            self.test_adj = []
            t_start = time.time()

            self.model_x.train()
            self.model_adj.train()

            wandb.log({"epoch": epoch})
            wandb.log({"lr_x": self.scheduler_x.get_lr()[0]})
            wandb.log({"lr_adj": self.scheduler_adj.get_lr()[0]})

            avg_loss_x = []
            avg_loss_adj = []
            avg_ep_time = []
            ep_time = 0
            steps = 0

            ep_st_time = time.time()
            for _, train_b in enumerate(self.train_loader):

                self.optimizer_x.zero_grad()
                self.optimizer_adj.zero_grad()
                x, adj = load_batch(train_b, self.device)
                loss_subject = (x, adj)

                loss_x, loss_adj = self.loss_fn(self.model_x, self.model_adj, *loss_subject)
                loss_x.backward()
                loss_adj.backward()

                torch.nn.utils.clip_grad_norm_(self.model_x.parameters(), self.config.train.grad_norm)
                torch.nn.utils.clip_grad_norm_(self.model_adj.parameters(), self.config.train.grad_norm)

                self.optimizer_x.step()
                self.optimizer_adj.step()

                # -------- EMA update --------
                self.ema_x.update(self.model_x.parameters())
                self.ema_adj.update(self.model_adj.parameters())

                loss_x_itm = loss_x.item()
                loss_adj_itm = loss_adj.item()
                self.train_x.append(loss_x_itm)
                self.train_adj.append(loss_adj_itm)
                avg_loss_x.append(loss_x_itm)
                avg_loss_adj.append(loss_adj_itm)

                if steps % self.config.train.step_log_interval == 0 and steps > 0:
                    wandb.log({"loss_x": np.mean(avg_loss_x)})
                    wandb.log({"loss_adj": np.mean(avg_loss_adj)})
                    avg_loss_x = []
                    avg_loss_adj = []

                steps += 1

            torch.cuda.synchronize()
            ep_time = time.time() - ep_st_time
            avg_ep_time.append(ep_time)
            wandb.log({"ep_time": ep_time})

            if self.config.train.lr_schedule:
                self.scheduler_x.step()
                self.scheduler_adj.step()

            self.model_x.eval()
            self.model_adj.eval()
            for _, test_b in enumerate(self.test_loader):

                x, adj = load_batch(test_b, self.device)
                loss_subject = (x, adj)

                with torch.no_grad():
                    self.ema_x.store(self.model_x.parameters())
                    self.ema_x.copy_to(self.model_x.parameters())
                    self.ema_adj.store(self.model_adj.parameters())
                    self.ema_adj.copy_to(self.model_adj.parameters())

                    loss_x, loss_adj = self.loss_fn(self.model_x, self.model_adj, *loss_subject)
                    self.test_x.append(loss_x.item())
                    self.test_adj.append(loss_adj.item())

                    self.ema_x.restore(self.model_x.parameters())
                    self.ema_adj.restore(self.model_adj.parameters())

            mean_train_x = np.mean(self.train_x)
            mean_train_adj = np.mean(self.train_adj)
            mean_test_x = np.mean(self.test_x)
            mean_test_adj = np.mean(self.test_adj)

            # -------- Log losses --------
            logger.log(f'{epoch+1:03d} | {time.time()-t_start:.2f}s | '
                        f'test x: {mean_test_x:.3e} | test adj: {mean_test_adj:.3e} | '
                        f'train x: {mean_train_x:.3e} | train adj: {mean_train_adj:.3e} | ', verbose=False)
            # -------- Log to wandb --------
            wandb.log({
                "train_x_ep_avg": mean_train_x,
                "train_adj_ep_avg": mean_train_adj,
                "test_x_ep_avg": mean_test_x,
                "test_adj_ep_avg": mean_test_adj,
            })

            # -------- Save checkpoints --------
            if epoch % self.config.train.save_interval == self.config.train.save_interval-1:
                save_name = f'_{epoch+1}' if epoch < self.config.train.num_epochs - 1 else ''

                torch.save({
                    'model_config': self.config,
                    'params_x' : self.params_x,
                    'params_adj' : self.params_adj,
                    'x_state_dict': self.model_x.state_dict(),
                    'adj_state_dict': self.model_adj.state_dict(),
                    'ema_x': self.ema_x.state_dict(),
                    'ema_adj': self.ema_adj.state_dict()
                    }, f'./checkpoints/{self.config.data.data}/{self.config.train.name}/{self.ckpt + save_name}.pth')

            # -------- Eval Generation metrics --------


            if epoch % self.config.train.print_interval == self.config.train.print_interval-1:
                tqdm.write(f'[EPOCH {epoch+1:04d}] test adj: {mean_test_adj:.3e} | train adj: {mean_train_adj:.3e} | '
                            f'test x: {mean_test_x:.3e} | train x: {mean_train_x:.3e}')

        torch.save({
            'model_config': self.config,
            'params_x' : self.params_x,
            'params_adj' : self.params_adj,
            'x_state_dict': self.model_x.state_dict(),
            'adj_state_dict': self.model_adj.state_dict(),
            'ema_x': self.ema_x.state_dict(),
            'ema_adj': self.ema_adj.state_dict()
            }, f'./checkpoints/{self.config.data.data}/{self.ckpt}.pth')

        wandb.log({"ep_time_avg": np.mean(avg_ep_time)})
        print(' ')
        return self.ckpt
