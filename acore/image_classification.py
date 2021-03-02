import numpy as np
import torch
import pickle
import math
from datetime import datetime
from cnn_models.alexnet import AlexNet, MLP
from cnn_models.resnet import ResNet18, ResNet34
from functools import partial


model_dict = {
    'mlp': MLP(num_classes=2),
    'alexnet': AlexNet(num_classes=2),
    'resnet18': ResNet18(num_classes=2),
    'resnet34': ResNet34(num_classes=2)
}


class ImgDataset(torch.utils.data.Dataset):

    def __init__(self, param_mat, image_mat, y_vec):
        self.param = param_mat
        self.image = torch.unsqueeze(image_mat, dim=1)
        self.y = y_vec

    def __getitem__(self, index):
        return self.param[index, :], self.image[index, :, :, :], self.y[index, :]

    def __len__(self):
        return self.param.shape[0]


def odds_loss_torch(out, y, true_tensor, false_tensor, p_param=0.5):

    mask_pos = torch.where((y == 1) & (out == out), true_tensor, false_tensor).bool()
    mask_neg = torch.where((y == 0) & (out == out), true_tensor, false_tensor).bool()

    out_pos = torch.masked_select(out, mask_pos).view(-1, 2)
    out_neg = torch.masked_select(out, mask_neg).view(-1, 2)

    return ((out_neg[:, 0] / out_neg[:, 1]) ** 2).mean() - 2 * (p_param / (1 - p_param)) * (
            out_pos[:, 0] / out_pos[:, 1]).mean()


def sample_from_prior(sample_size, random_seed):
    np.random.seed(random_seed)
    alpha_prior_sample = np.random.uniform(-math.pi, math.pi, size=sample_size)
    lambda_prior_sample = np.random.uniform(0, 1, size=sample_size)
    return np.hstack((alpha_prior_sample.reshape(-1, 1), lambda_prior_sample.reshape(-1, 1)))


def setup_img_data(param_mat, image_mat, midpoint, seed=7):

    # Sample from the simulator
    param_simulator = param_mat[:midpoint, :]
    image_simulator = image_mat[:midpoint, :, :]
    y_vec_simulator = np.hstack((np.zeros(midpoint).reshape(-1, 1), np.ones(midpoint).reshape(-1, 1)))

    # Sample from the empirical marginal
    param_empirical_marginal = sample_from_prior(sample_size=midpoint, random_seed=seed)
    image_empirical_marginal = image_mat[midpoint:(2 * midpoint), :, :]
    y_vec_empirical_marginal = np.hstack((np.ones(midpoint).reshape(-1, 1), np.zeros(midpoint).reshape(-1, 1)))

    # Create full matrices and vectors
    param_full = np.vstack((param_simulator, param_empirical_marginal))
    image_full = np.vstack((image_simulator, image_empirical_marginal))
    y_vec_full = np.vstack((y_vec_simulator, y_vec_empirical_marginal))

    return param_full, image_full, y_vec_full


def main(datafile='galsim/acore_galsim_simulated_275000params_1ssobs_downsampling20_0.5mixingparam_2021-02-08-02-37.pkl',
         img_h=20, img_w=20, seed=7, n_train=5000, n_check=5000, test_split=0.1,
         cuda_flag=False, batch_size=512, lr=1e-6, lr_decay=False, epochs=10000, epochs_check=25,
         patience_early_stopping=5, model_run='mlp'):

    # Load the images
    img_dict = pickle.load(open('data/' + datafile, 'rb'))

    # Prepare the arrays
    n_images = len(img_dict.keys())
    param_mat = np.zeros((n_images, 2))
    image_mat = np.zeros((n_images, img_h, img_w))
    idx = 0
    for (alpha_val, lambda_val), image_np in img_dict.items():

        if idx > ((n_train * 2 + n_check) + 1):
            break

        # Check whether the image was available
        if len(image_np) > 0:
            param_mat[idx, :] = np.array([alpha_val, lambda_val])
            image_mat[idx, :, :] = image_np[0]
            idx += 1

    # Correct the shape of parameter and image matrices
    n_images = idx
    param_mat = param_mat[:n_images, :]
    image_mat = image_mat[:n_images, :, :]

    # Generate the samples from the simulator and the sample from the empirical marginal
    if n_train > (n_images/2):
        raise ValueError('Not enough training data. Current shape:%d, Requested:%d (x2).' % (n_images, n_train))

    # Setup the sample for training
    param_full, image_full, y_vec_full = setup_img_data(
        param_mat[:(2 * n_train), :], image_mat[:(2 * n_train), :, :], midpoint=n_train)
    param_check, image_check, y_check = setup_img_data(
        param_mat[(2 * n_train):(2 * n_train + n_check), :],
        image_mat[(2 * n_train):(2 * n_train + n_check), :, :], midpoint=int(n_check/2))

    # Split training and testing
    np.random.seed(seed)
    indices = np.random.permutation(n_train * 2)
    test_n = int((n_train * 2) * test_split)
    test_idx, train_idx = indices[:test_n], indices[test_n:]
    train_param, train_img, train_y = param_full[train_idx, :], image_full[train_idx, :, :], y_vec_full[train_idx]
    test_param, test_img, test_y = param_full[test_idx, :], image_full[test_idx, :, :], y_vec_full[test_idx]

    # Assign it to the device
    device = torch.device('cuda:0' if (torch.cuda.is_available() and cuda_flag) else 'cpu')

    train_dataset = ImgDataset(
        param_mat=torch.from_numpy(train_param.astype(np.float64)).type(torch.Tensor),
        image_mat=torch.from_numpy(train_img.astype(np.float64)).type(torch.Tensor),
        y_vec=torch.from_numpy(train_y.astype(np.float64)).type(torch.Tensor))
    test_dataset = ImgDataset(
        param_mat=torch.from_numpy(test_param.astype(np.float64)).type(torch.Tensor),
        image_mat=torch.from_numpy(test_img.astype(np.float64)).type(torch.Tensor),
        y_vec=torch.from_numpy(test_y.astype(np.float64)).type(torch.Tensor))
    check_dataset = ImgDataset(
        param_mat=torch.from_numpy(param_check.astype(np.float64)).type(torch.Tensor),
        image_mat=torch.from_numpy(image_check.astype(np.float64)).type(torch.Tensor),
        y_vec=torch.from_numpy(y_check.astype(np.float64)).type(torch.Tensor))

    train_load = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
    test_load = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=test_n, shuffle=False)
    check_load = torch.utils.data.DataLoader(dataset=check_dataset, batch_size=n_check, shuffle=False)

    # Create the model
    flnm_model = 'data/%smodel_%slr_%sbatchsize_%s.pt' % (
        model_run, str(lr), str(batch_size), datetime.strftime(datetime.today(), '%Y-%m-%d'))
    model = model_dict[model_run]
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCELoss(reduction='mean').to(device)
    or_loss = partial(odds_loss_torch,
                      true_tensor=torch.tensor([1]).to(device), false_tensor=torch.tensor([0]).to(device))

    # Creating a scheduler in case the LR decay is passed. In case the LR decay is 0, then the decay is set to happen
    # after the last epoch (so it's equivalent to not happening)
    step_size = lr_decay if lr_decay > 0 else epochs + 1
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)

    # Start the training
    training_loss, test_loss, check_loss = [], [], []
    odds_loss_training, odds_loss_test, odds_loss_check = [], [], []
    patience_counter = 0
    for epoch in range(epochs):

        if patience_counter > patience_early_stopping:
            break

        model.train()
        train_loss_temp = []
        odds_loss_temp = []
        for batch_idx, (param_batch, img_batch, y_batch) in enumerate(train_load):
            param_batch, img_batch, y_batch = param_batch.to(device), img_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            out_batch = model(img_batch, param_batch)
            loss = criterion(out_batch, y_batch)
            loss.backward()
            optimizer.step()
            train_loss_temp.append(loss.item())
            odds_loss_temp.append(or_loss(out_batch, y_batch).item())

        scheduler.step()
        weights_average = np.ones(len(train_loss_temp))
        weights_average[-1] = (train_param.shape[0] % batch_size) / batch_size
        training_loss.append(np.average(train_loss_temp, weights=weights_average))
        odds_loss_training.append(np.average(odds_loss_temp, weights=weights_average))

        model.eval()
        param_batch_test, img_batch_test, y_batch_test = next(iter(test_load))
        param_batch_test, img_batch_test, y_batch_test = param_batch_test.to(device), \
                                                         img_batch_test.to(device), y_batch_test.to(device)
        out_batch_test = model(img_batch_test, param_batch_test)
        acc_test = (torch.argmax(out_batch_test, dim=1) == torch.argmax(y_batch_test, dim=1)).sum()/test_n
        test_loss.append(criterion(out_batch_test, y_batch_test).item())
        odds_loss_test.append(or_loss(out_batch_test, y_batch_test).item())

        param_batch_check, img_batch_check, y_batch_check = next(iter(check_load))
        param_batch_check, img_batch_check, y_batch_check = param_batch_check.to(device), \
                                                         img_batch_check.to(device), y_batch_check.to(device)
        out_batch_check = model(img_batch_check, param_batch_check)
        check_loss.append(criterion(out_batch_check, y_batch_check).item())
        odds_loss_check.append(or_loss(out_batch_check, y_batch_check).item())

        if epoch == epochs_check:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'training_loss': training_loss,
                'test_loss': test_loss,
                'odds_loss_training': odds_loss_training,
                'odds_loss_test': odds_loss_test,
                'check_loss': check_loss,
                'odds_loss_check': odds_loss_check
            }, flnm_model)

        if epoch % epochs_check == 0:
            print('Epoch:%d, Train CE: %.4f, Train OL: %.4f,'
                  ' Test Loss: %.4f, Test OL: %.4f, Valid CE:%.4f, Valid OL:%.4f Acc: %.4f' % (
                      epoch, training_loss[-1], odds_loss_training[-1],
                      test_loss[-1], odds_loss_test[-1], check_loss[-1], odds_loss_check[-1], acc_test))

            if epoch > (2 * epochs_check) and np.median(
                    test_loss[-epochs_check:]) > np.median(test_loss[-(2 * epochs_check):-epochs_check]):
                patience_counter += 1
            else:
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'training_loss': training_loss,
                    'test_loss': test_loss,
                    'odds_loss_training': odds_loss_training,
                    'odds_loss_test': odds_loss_test,
                    'check_loss': check_loss,
                    'odds_loss_check': odds_loss_check
                }, flnm_model)


if __name__ == '__main__':

    main()