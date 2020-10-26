import torch
import torch.nn as nn
import numpy as np


class Dataset(torch.utils.data.Dataset):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


def weights_init(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # apply a uniform distribution to the weights and a bias=0
        m.weight.data.uniform_(0.01, 0.02)
        m.bias.data.fill_(0.0)


def or_loss(out, y, true_tensor, false_tensor, p_param=0.5):
    y_mat = y.unsqueeze(1).repeat(1, 2)

    mask_pos = torch.where((y_mat == 1) & (out == out), true_tensor, false_tensor).bool()
    mask_neg = torch.where((y_mat == 0) & (out == out), true_tensor, false_tensor).bool()

    out_pos = torch.masked_select(out, mask_pos).view(-1, 2)
    out_neg = torch.masked_select(out, mask_neg).view(-1, 2)

    return ((out_neg[:, 0] / out_neg[:, 1]) ** 2).mean() - 2 * (p_param / (1 - p_param)) * (
                out_pos[:, 0] / out_pos[:, 1]).mean()


def or_loss_singleodds(out, y, true_tensor, false_tensor, p_param=0.5):
    mask_pos = torch.where((y == 1), true_tensor, false_tensor).bool()
    mask_neg = torch.where((y == 0), true_tensor, false_tensor).bool()

    out_pos = torch.masked_select(out, mask_pos).view(-1, 1)
    out_neg = torch.masked_select(out, mask_neg).view(-1, 1)

    return (out_neg ** 2).mean() - 2 * (p_param / (1 - p_param)) * out_pos.mean()


class OddsNet(nn.Sequential):

    def __init__(self, direct_odds=False, layers=(100,), batch_size=64, n_epochs=25000,
                 epoch_check=250, learning_rate=1e-4, precision=1e-6, verbose=False):
        super().__init__()

        self.layers = layers
        self.direct_odds = direct_odds
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.true_tensor = torch.tensor([1]).to(self.device)
        self.false_tensor = torch.tensor([0]).to(self.device)
        self.n_epochs = n_epochs
        self.epoch_check = epoch_check
        self.custom_loss = or_loss_singleodds if self.direct_odds else or_loss
        self.learning_rate = learning_rate
        self.precision = precision
        self.verbose = verbose

    def forward(self, x):
        return super().forward(x).squeeze()

    def fit(self, X, y):

        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        input_size = X.shape[1]

        # Create the actual net now
        for idx, layer_val in enumerate(self.layers):
            if idx == 0:
                self.add_module('lin%s' % idx, nn.Linear(input_size, layer_val))
            else:
                self.add_module('lin%s' % idx, nn.Linear(self.layers[idx - 1], layer_val))
            self.add_module("ReLU", nn.ReLU())
        if self.direct_odds:
            self.add_module('lin%s' % len(self.layers), nn.Linear(self.layers[-1], 1))
        else:
            self.add_module('lin%s' % len(self.layers), nn.Linear(self.layers[-1], 2))
            self.add_module('softmax', nn.Softmax(dim=1))

        # Initialize the weights randomly
        self.apply(weights_init)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        # Add the data to the device
        x = torch.from_numpy(X.astype(np.float64)).type(torch.Tensor).to(self.device)
        y = torch.from_numpy(y.astype(np.float64)).type(torch.Tensor).to(self.device)
        train_load = torch.utils.data.DataLoader(dataset=Dataset(x, y), batch_size=self.batch_size, shuffle=False)

        # Training the actual net
        loss_list = []
        loss_list_check = []
        for epoch in range(self.n_epochs):
            self.train()
            for batch_idx, (x_batch, y_basis_batch) in enumerate(train_load):
                x_batch, y_basis_batch = x_batch.to(self.device), y_basis_batch.to(self.device)
                optimizer.zero_grad()
                out_batch = self.forward(x_batch)
                loss = self.custom_loss(out_batch, y_basis_batch,
                                        true_tensor=self.true_tensor, false_tensor=self.false_tensor)

                loss.backward()
                optimizer.step()

                loss_list.append(loss.item())

            if epoch % self.epoch_check == 0:
                if self.verbose:
                    print('Epoch %d, Training Loss: %.8f' % (epoch, loss_list[-1]))
                loss_list_check.append(loss_list[-1])
                if len(loss_list_check) > 2 and (loss_list_check[-3] - loss_list_check[-1]) <= self.precision:
                    break

    def predict_proba(self, X):
        x_test = torch.from_numpy(X.astype(np.float64)).type(torch.Tensor).to(self.device)
        out_test = self.forward(x_test).detach().numpy()

        if self.direct_odds:
            out_mat = np.hstack((np.ones((out_test.shape[0], 1)), out_test.reshape(-1, 1)))
        else:
            out_mat = out_test

        return out_mat
