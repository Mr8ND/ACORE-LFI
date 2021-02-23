import numpy as np
import torch
import pickle
import math
from datetime import datetime
from cnn_models.alexnet import AlexNet, MLP
from cnn_models.resnet import ResNet18, ResNet34


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


def sample_from_prior(sample_size, random_seed):
    np.random.seed(random_seed)
    alpha_prior_sample = np.random.uniform(-math.pi, math.pi, size=sample_size)
    lambda_prior_sample = np.random.uniform(0, 1, size=sample_size)
    return np.hstack((alpha_prior_sample.reshape(-1, 1), lambda_prior_sample.reshape(-1, 1)))


def main(datafile='acore_galsim_simulated_275000params_1ssobs_downsampling20_0.5mixingparam_2021-02-08-02-37.pkl',
         img_h=20, img_w=20, seed=7, n_train=10000, test_split=0.1, cuda_flag=False, batch_size=512, lr=1e-6,
         lr_decay=False, epochs=10000, epochs_check=1, model_run='resnet34'):

    # Load the images
    img_dict = pickle.load(open('data/' + datafile, 'rb'))

    # Prepare the arrays
    n_images = len(img_dict.keys())
    param_mat = np.zeros((n_images, 2))
    image_mat = np.zeros((n_images, img_h, img_w))
    idx = 0
    for (alpha_val, lambda_val), image_np in img_dict.items():

        if idx > ((n_train * 2) + 1):
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

    # Sample from the simulator
    param_simulator = param_mat[:n_train, :]
    image_simulator = image_mat[:n_train, :, :]
    y_vec_simulator = np.hstack((np.zeros(n_train).reshape(-1, 1), np.ones(n_train).reshape(-1, 1)))

    # Sample from the empirical marginal
    param_empirical_marginal = sample_from_prior(sample_size=n_train, random_seed=seed)
    image_empirical_marginal = image_mat[n_train:(2 * n_train), :, :]
    y_vec_empirical_marginal = np.hstack((np.ones(n_train).reshape(-1, 1), np.zeros(n_train).reshape(-1, 1)))

    # Create full matrices and vectors
    param_full = np.vstack((param_simulator, param_empirical_marginal))
    image_full = np.vstack((image_simulator, image_empirical_marginal))
    y_vec_full = np.vstack((y_vec_simulator, y_vec_empirical_marginal))

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

    train_load = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
    test_load = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=test_n, shuffle=False)

    # Create the model
    flnm_model = 'data/%smodel_%slr_%sbatchsize_%s.pt' % (
        model_run, str(lr), str(batch_size), datetime.strftime(datetime.today(), '%Y-%m-%d'))
    model = model_dict[model_run]
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCELoss(reduction='mean').to(device)

    # Creating a scheduler in case the LR decay is passed. In case the LR decay is 0, then the decay is set to happen
    # after the last epoch (so it's equivalent to not happening)
    step_size = lr_decay if lr_decay > 0 else epochs + 1
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)

    # Start the training
    training_loss = []
    test_loss = []

    for epoch in range(epochs):
        model.train()
        train_loss_temp = []
        for batch_idx, (param_batch, img_batch, y_batch) in enumerate(train_load):
            param_batch, img_batch, y_batch = param_batch.to(device), img_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            out_batch = model(img_batch, param_batch)
            loss = criterion(out_batch, y_batch)
            loss.backward()
            optimizer.step()
            train_loss_temp.append(loss.item())

        scheduler.step()
        weights_average = np.ones(len(train_loss_temp))
        weights_average[-1] = (train_param.shape[0] % batch_size) / batch_size
        training_loss.append(np.average(train_loss_temp, weights=weights_average))

        model.eval()
        param_batch_test, img_batch_test, y_batch_test = next(iter(test_load))
        param_batch_test, img_batch_test, y_batch_test = param_batch_test.to(device), \
                                                         img_batch_test.to(device), y_batch_test.to(device)
        out_batch_test = model(img_batch_test, param_batch_test)
        acc_test = (torch.argmax(out_batch_test, dim=1) == torch.argmax(y_batch_test, dim=1)).sum()/test_n
        loss = criterion(out_batch_test, y_batch_test)
        test_loss.append(loss.item())

        if epoch % epochs_check == 0:
            print('Epoch:%d, Train Loss: %.5f, Test Loss: %.5f, Accuracy Test: %.5f' % (
                epoch, training_loss[-1], test_loss[-1], acc_test))

        if epoch > 200 and np.median(
                training_loss[-100:]) > np.median(training_loss[-200:-100]):
            break
        else:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
                'training_loss': training_loss,
                'test_loss': test_loss
            }, flnm_model)



if __name__ == '__main__':

    main()