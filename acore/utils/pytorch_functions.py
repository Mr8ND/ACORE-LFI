import torch
import torch.nn as nn
import numpy as np
from itertools import chain

# Quantile regression code taken from https://colab.research.google.com/drive/1nXOlrmVHqCHiixqiMF6H8LSciz583_W2


class q_model(nn.Module):
    def __init__(self,
                 quantiles,
                 neur_shapes,
                 in_shape=1,
                 dropout=0.5,
                 seed=7):
        super().__init__()
        self.quantiles = quantiles
        self.num_quantiles = len(quantiles)
        self.neur_shapes = neur_shapes
        self.in_shape = in_shape
        self.seed = seed
        self.out_shape = len(quantiles)
        self.dropout = dropout
        self.build_model()
        self.init_weights()

    def build_model(self):
        self.base_model = nn.Sequential(
            nn.Linear(self.in_shape, self.neur_shapes[0]),
            nn.ReLU(),
            # nn.BatchNorm1d(64),
            nn.Dropout(self.dropout),
            nn.Linear(self.neur_shapes[0], self.neur_shapes[1]),
            nn.ReLU(),
            # nn.BatchNorm1d(64),
            nn.Dropout(self.dropout),
        )
        final_layers = [
            nn.Linear(self.neur_shapes[1], 1) for _ in range(len(self.quantiles))
        ]
        self.final_layers = nn.ModuleList(final_layers)

    def init_weights(self):
        torch.manual_seed(self.seed)
        for m in chain(self.base_model, self.final_layers):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        tmp_ = self.base_model(x)
        return torch.cat([layer(tmp_) for layer in self.final_layers], dim=1)


class q_model_3l(nn.Module):
    def __init__(self,
                 quantiles,
                 neur_shapes,
                 seed=7,
                 in_shape=1,
                 dropout=0.5):
        super().__init__()
        self.quantiles = quantiles
        self.num_quantiles = len(quantiles)
        self.neur_shapes = neur_shapes
        self.in_shape = in_shape
        self.seed = seed
        self.out_shape = len(quantiles)
        self.dropout = dropout
        self.build_model()
        self.init_weights()

    def build_model(self):
        self.base_model = nn.Sequential(
            nn.Linear(self.in_shape, self.neur_shapes[0]),
            nn.ReLU(),
            # nn.BatchNorm1d(64),
            nn.Dropout(self.dropout),
            nn.Linear(self.neur_shapes[0], self.neur_shapes[1]),
            nn.ReLU(),
            # nn.BatchNorm1d(64),
            nn.Linear(self.neur_shapes[1], self.neur_shapes[2]),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )
        final_layers = [
            nn.Linear(self.neur_shapes[-1], 1) for _ in range(len(self.quantiles))
        ]
        self.final_layers = nn.ModuleList(final_layers)

    def init_weights(self):
        torch.manual_seed(self.seed)
        for m in chain(self.base_model, self.final_layers):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        tmp_ = self.base_model(x)
        return torch.cat([layer(tmp_) for layer in self.final_layers], dim=1)


class QuantileLoss(nn.Module):
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - preds[:, i]
            losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(1))
        loss = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))
        return loss


class Learner:
    def __init__(self, model, optimizer_class, loss_func, device='cpu', seed=7):
        self.model = model.to(device)
        self.optimizer = optimizer_class(self.model.parameters())
        self.loss_func = loss_func.to(device)
        self.device = device
        self.seed = seed
        self.loss_history = []

    def fit(self, x, y, epochs, batch_size):
        torch.manual_seed(self.seed)
        self.model.train()
        for e in range(epochs):
            shuffle_idx = np.arange(x.shape[0])
            np.random.shuffle(shuffle_idx)
            x = x[shuffle_idx]
            y = y[shuffle_idx]
            epoch_losses = []
            for idx in range(0, x.shape[0], batch_size):
                self.optimizer.zero_grad()
                batch_x = torch.from_numpy(
                    x[idx: min(idx + batch_size, x.shape[0]), :]
                ).float().to(self.device).requires_grad_(False)
                batch_y = torch.from_numpy(
                    y[idx: min(idx + batch_size, y.shape[0])]
                ).float().to(self.device).requires_grad_(False)
                preds = self.model(batch_x)
                loss = self.loss_func(preds, batch_y)
                loss.backward()
                self.optimizer.step()
                epoch_losses.append(loss.cpu().detach().numpy())
            epoch_loss = np.mean(epoch_losses)
            self.loss_history.append(epoch_loss)

    def predict(self, x, mc=False):
        if mc:
            self.model.train()
        else:
            self.model.eval()
        return self.model(torch.from_numpy(x).to(self.device).requires_grad_(False)).cpu().detach().numpy()