from typing import Optional, Mapping
from collections import deque

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class Model(object):
    def __init__(self, network, gradient_clip_value=5.0, **kwargs):
        self.model = network(**kwargs).cuda()
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.state = {}
        self.gradient_clip_value, self.gradient_norm_queue = gradient_clip_value, deque([np.inf], maxlen=5)
        self.optimizer = None

    def train_step(self, train_x: torch.Tensor, train_y: torch.Tensor):
        self.optimizer.zero_grad()
        self.model.train()
        scores = self.model(train_x)
        loss = self.loss_fn(scores, train_y)
        loss.backward()
        self.clip_gradient()
        self.optimizer.step(closure=None)
        return loss.item()

    def predict_step(self, data_x: torch.Tensor, k: int):
        self.model.eval()
        with torch.no_grad():
            if isinstance(data_x, dict):
                data_x = {k:v.cuda() for k,v in data_x.items()}
            else:
                data_x = data_x.cuda()
            scores, labels = torch.topk(self.model(data_x), k)
            return torch.sigmoid(scores).cpu(), labels.cpu()

    def get_optimizer(self, optimizer=None, **kwargs):
        if optimizer is None:
            self.optimizer = DenseSparseAdam(self.model.parameters(), **kwargs)
        else:
            self.optimizer = optimizer(self.model.parameters(), **kwargs)

    def train(self, train_loader: DataLoader, valid_loader: DataLoader, opt_params: Optional[Mapping] = None,
              nb_epoch=100, step=100, k=5, early=100, verbose=True, swa_warmup=None, optimizer=None, **kwargs):
        self.get_optimizer(**({} if opt_params is None else opt_params))
        global_step, best_n5, e = 0, 0.0, 0
        print_loss = 0.0
        for epoch_idx in range(nb_epoch):
            if epoch_idx == swa_warmup:
                self.swa_init()
            for i, (train_x, train_y) in enumerate(train_loader, 1):
                global_step += 1
                if isinstance(train_x, dict):
                    train_x = {k:v.cuda() for k,v in train_x.items()}
                else:
                    train_x = train_x.cuda()
                train_y = train_y.cuda()
                loss = self.train_step(train_x, train_y)
                print_loss += loss#
                if global_step % step == 0:
                    self.swa_step()
                    self.swap_swa_params()
                    ##
                    labels = []
                    valid_loss = 0.0
                    self.model.eval()
                    with torch.no_grad():
                        for (valid_x, valid_y) in valid_loader:
                            if isinstance(valid_x, dict):
                                valid_x = {k:v.cuda() for k,v in valid_x.items()}
                            else:
                                valid_x = valid_x.cuda()
                            valid_y = valid_y.cuda()
                            logits = self.model(valid_x)
                            valid_loss += self.loss_fn(logits, valid_y.cuda()).item()
                            scores, tmp = torch.topk(logits, k)
                            labels.append(tmp.cpu())
                    valid_loss /= len(valid_loader)
                    labels = np.concatenate(labels)
                    targets = valid_loader.dataset.data_y
                    p5, n5 = get_p_5(labels, targets), get_n_5(labels, targets)
                    self.swap_swa_params()
                    if verbose:
                        log_msg = '%d %d train loss: %.7f valid loss: %.7f P@5: %.5f N@5: %.5f early stop: %d' % \
                        (epoch_idx, i * train_loader.batch_size, print_loss / step, valid_loss, round(p5, 5), round(n5, 5), e)
                        print(log_msg)
                        print_loss = 0.0

    def predict(self, data_loader: DataLoader, k=100, desc='Predict', **kwargs):
        scores_list, labels_list = zip(*(self.predict_step(data_x, k)
                                         for data_x, *_ in tqdm(data_loader, desc=desc, leave=False)))
        return np.concatenate(scores_list), np.concatenate(labels_list)

    def clip_gradient(self):
        if self.gradient_clip_value is not None:
            max_norm = max(self.gradient_norm_queue)
            total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm * self.gradient_clip_value)
            self.gradient_norm_queue.append(min(total_norm, max_norm * 2.0, 1.0))

    def swa_init(self):
        if 'swa' not in self.state:
            print('SWA Initializing')
            swa_state = self.state['swa'] = {'models_num': 1}
            for n, p in self.model.named_parameters():
                swa_state[n] = p.data.cpu().detach()

    def swa_step(self):
        if 'swa' in self.state:
            swa_state = self.state['swa']
            swa_state['models_num'] += 1
            beta = 1.0 / swa_state['models_num']
            with torch.no_grad():
                for n, p in self.model.named_parameters():
                    swa_state[n].mul_(1.0 - beta).add_(beta, p.data.cpu())

    def swap_swa_params(self):
        if 'swa' in self.state:
            swa_state = self.state['swa']
            for n, p in self.model.named_parameters():
                p.data, swa_state[n] = swa_state[n].cuda(), p.data.cpu()

    def disable_swa(self):
        if 'swa' in self.state:
            del self.state['swa']
