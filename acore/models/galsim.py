import pickle
import numpy as np
import math
import os
import torch
import sys
import pdb
import random
sys.path.append(".")

from cnn_models.alexnet import AlexNet, MLP
from cnn_models.resnet import ResNet18, ResNet34


MODEL_DICT = {
    'mlp': MLP,
    'alexnet': AlexNet,
    'resnet18': ResNet18,
    'resnet34': ResNet34
}


class CNNmodel:

    def __init__(self, model_name, model_folder, d, img_h, img_w, pretrained=True, cuda_flag=False):
        self.model_name = model_name
        self.device = torch.device('cuda:0' if (torch.cuda.is_available() and cuda_flag) else 'cpu')
        self.model = MODEL_DICT[model_name]().to(self.device)
        self.model_folder = model_folder
        self.d = d
        self.img_h = img_h
        self.img_w = img_w

        self.check_loss = None
        self.odds_loss_check = None

        if pretrained:
            self._load_torch_model(model_name)

    def _load_torch_model(self, model_name):
        flnm_model = [el for el in os.listdir(self.model_folder) if model_name in el][0]
        model_out_dict = torch.load(self.model_folder + flnm_model, map_location=self.device)
        self.model.load_state_dict(model_out_dict['model_state_dict'])
        self.check_loss = model_out_dict['check_loss']
        self.odds_loss_check = model_out_dict['odds_loss_check']

    def predict_proba(self, predict_mat):
        param_tensor = torch.from_numpy(predict_mat[:, :self.d]).type(torch.Tensor).to(self.device)
        img_tensor = torch.from_numpy(
            predict_mat[:, self.d:].reshape(-1, self.img_h, self.img_w)).type(torch.Tensor)
        img_tensor = img_tensor.unsqueeze(1).to(self.device)

        return self.model(img=img_tensor, param=param_tensor).cpu().detach().numpy()


class GalSimLoader:
    # Simulations are stored in a dictionary, where the keys are the parameter values and the values are the
    # images.

    def __init__(self, model_name='resnet34', model_folder='cnn_models/trained_models/',
                 flnm_sim='data/galsim/acore_galsim_simulated_50000params_25ssobs_downsampling20_0.5mixingparam_2021-02-08-16-50.pkl',
                 flnm_data='data/galsim/acore_galsim_simulated_central_param_100ssobs_downsampling20_0.5mixingparam_2021-02-06-18-53.pkl',
                 flnm_train='data/galsim/acore_galsim_simulated_275000params_1ssobs_downsampling20_0.5mixingparam_2021-02-08-02-37.pkl',
                 alpha_low=-math.pi, alpha_high=math.pi, lambda_low=0, lambda_high=1, alpha_true=0.0, lambda_true=0.5,
                 out_dir='galsim/', num_acore_grid=21, num_pred_grid=21, seed=7, cuda_flag=False, *args, **kwargs):

        # Set the seed
        self.seed = seed
        random.seed(self.seed)

        # Set all parameters
        self.d = 2
        self.img_h = 20
        self.img_w = 20
        self.d_obs = self.img_h * self.img_w
        self.out_directory = out_dir
        self.b_prime_vec = [100, 500, 1000, 5000, 10000, 50000]
        self.true_param = np.array([alpha_true, lambda_true])
        self.alpha_low = alpha_low
        self.alpha_high = alpha_high
        self.lambda_low = lambda_low
        self.lambda_high = lambda_high

        # Create the prediction and grid over which to minimize ACORE
        self.pred_grid = np.dstack(np.meshgrid(
            np.linspace(start=self.alpha_low, stop=self.alpha_high, num=num_pred_grid),
            np.linspace(start=self.lambda_low, stop=self.lambda_high, num=num_pred_grid)
        )).reshape(-1, 2)
        self.idx_row_true_param = np.where((self.pred_grid == self.true_param).all(axis=1))[0][0]
        self.acore_grid = np.dstack(np.meshgrid(
            np.linspace(start=self.alpha_low, stop=self.alpha_high, num=num_acore_grid),
            np.linspace(start=self.lambda_low, stop=self.lambda_high, num=num_acore_grid)
        )).reshape(-1, 2)

        # Load in the data for MSNH and the true observed data
        self.flnm_train = flnm_train
        self.flnm_sim = flnm_sim
        self.load_simulated_images()
        self.param_mshn_trueobs_dict = pickle.load(open(flnm_data, 'rb'))

        # Check if parameters correspond
        if not np.array_equal(np.array(list(self.param_mshn_trueobs_dict.keys())[0]), self.true_param):
            raise ValueError('True parameters are not what was expected.')

        # Load pre-trained model
        self.clf_obj = CNNmodel(model_name=model_name, model_folder=model_folder, d=self.d,
                                img_h=self.img_h, img_w=self.img_w, cuda_flag=cuda_flag)

    def load_simulated_images(self):
        param_mshn_dict_temp = pickle.load(open(self.flnm_sim, 'rb'))
        self.param_mshn_dict = {k: v for k, v in param_mshn_dict_temp.items() if len(v) > 0}
        self.param_vec = list(self.param_mshn_dict.keys())

    def sample_param_values(self, sample_size):
        alpha_prior_sample = np.random.uniform(self.alpha_low, self.alpha_high, size=sample_size)
        lambda_prior_sample = np.random.uniform(self.lambda_low, self.lambda_high, size=sample_size)
        return np.hstack((alpha_prior_sample.reshape(-1, 1), lambda_prior_sample.reshape(-1, 1)))

    def sample_sim_check(self, sample_size, n):
        param_mat = np.zeros((n, self.d))
        sample_mat = np.zeros((n, sample_size, self.d_obs))

        for kk in range(n):
            param_val = random.sample(self.param_vec, 1)[0]
            param_mat[kk, :] = np.array([param_val[0], param_val[1]])
            ss_temp = 0
            while ss_temp < sample_size:
                try:
                    sample_img = self.param_mshn_dict[param_val].pop()
                except IndexError:
                    self.load_simulated_images()
                    continue
                sample_mat[kk, ss_temp, :] = sample_img.reshape(self.d_obs,)
                ss_temp += 1

        return param_mat, sample_mat

    def sample_sim_true_param(self, sample_size):
        sample_mat = np.zeros((sample_size, self.d_obs))
        true_param_tuple = list(self.param_mshn_trueobs_dict.keys())[0]
        ss_temp = 0
        while ss_temp < sample_size:
            sample_img = self.param_mshn_trueobs_dict[true_param_tuple].pop()
            sample_mat[ss_temp, :] = sample_img.reshape(self.d_obs, )
            ss_temp += 1

        return sample_mat

    def sample_msnh_algo5(self, b_prime, sample_size):
        return self.sample_sim_check(sample_size=sample_size, n=b_prime)

    def _sample_from_prior_training(self, sample_size, random_seed):
        np.random.seed(random_seed)
        alpha_prior_sample = np.random.uniform(-math.pi, math.pi, size=sample_size)
        lambda_prior_sample = np.random.uniform(0, 1, size=sample_size)
        return np.hstack((alpha_prior_sample.reshape(-1, 1), lambda_prior_sample.reshape(-1, 1)))

    def load_test_cases(self, n_train, test_split=0.1):

        # Load the images
        img_dict = pickle.load(open(self.flnm_train, 'rb'))

        # Prepare the arrays
        n_images = len(img_dict.keys())
        param_mat = np.zeros((n_images, 2))
        image_mat = np.zeros((n_images, self.img_h, self.img_w))
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
        if n_train > (n_images / 2):
            raise ValueError('Not enough training data. Current shape:%d, Requested:%d (x2).' % (n_images, n_train))

        # Sample from the simulator
        param_simulator = param_mat[:n_train, :]
        image_simulator = image_mat[:n_train, :, :]
        y_vec_simulator = np.hstack((np.zeros(n_train).reshape(-1, 1), np.ones(n_train).reshape(-1, 1)))

        # Sample from the empirical marginal
        param_empirical_marginal = self._sample_from_prior_training(sample_size=n_train, random_seed=self.seed)
        image_empirical_marginal = image_mat[n_train:(2 * n_train), :, :]
        y_vec_empirical_marginal = np.hstack((np.ones(n_train).reshape(-1, 1), np.zeros(n_train).reshape(-1, 1)))

        # Create full matrices and vectors
        param_full = np.vstack((param_simulator, param_empirical_marginal))
        image_full = np.vstack((image_simulator, image_empirical_marginal))
        y_vec_full = np.vstack((y_vec_simulator, y_vec_empirical_marginal))

        # Split training and testing
        np.random.seed(self.seed)
        indices = np.random.permutation(n_train * 2)
        test_n = int((n_train * 2) * test_split)
        test_idx = indices[:test_n]
        y_test_full = y_vec_full[test_idx]

        # Have to flatten the y_test to make it Sklearn compatible
        y_test_1d = np.array([int(el[1] == 1) for el in y_test_full]).reshape(-1, )

        return image_full[test_idx, :, :], param_full[test_idx, :], y_test_1d

    def compute_exact_tau(self, x_obs, t0_val, meshgrid):
        raise NotImplementedError('True Likelihood not known for this model.')

    def compute_exact_tau_distr(self, t0_val, meshgrid, n_sampled, sample_size_obs):
        raise NotImplementedError('True Likelihood not known for this model.')

    def make_grid_over_param_space(self, n_grid):
        raise NotImplementedError('Grid is fixed on this model for now.')
