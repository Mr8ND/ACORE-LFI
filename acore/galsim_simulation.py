import galsim
import argparse
import numpy as np
import math
import coord
import pickle
from datetime import datetime
from tqdm.auto import tqdm


def downsample(myarr, factor, estimator=np.nanmean):
    """
    Downsample a 2D array by averaging over *factor* pixels in each axis.
    Crops upper edge if the shape is not a multiple of factor.
    This code is pure np and should be fast.
    keywords:
        estimator - default to mean.  You can downsample by summing or
            something else if you want a different estimator
            (e.g., downsampling error: you want to sum & divide by sqrt(n))
    """
    ys, xs = myarr.shape
    crarr = myarr[:ys - (ys % int(factor)), :xs - (xs % int(factor))]
    dsarr = estimator(np.concatenate([[crarr[i::factor, j::factor]
                                       for i in range(factor)]
                                      for j in range(factor)]), axis=0)
    return dsarr


def sample_from_prior(sample_size, main_peak):
    alpha_prior_sample = np.random.uniform(-math.pi, math.pi, size=sample_size)
    # lambda_prior_sample = np.random.uniform(0, 1, size=sample_size)
    bern_sample = np.random.binomial(n=1, p=0.1, size=sample_size)
    lambda_prior_sample = np.array([main_peak, 0.1])[bern_sample]

    return alpha_prior_sample, lambda_prior_sample


def sample_true_values(alpha_prior_sample, lambda_prior_sample, main_peak, mixing_param=0.5):

    assert alpha_prior_sample.shape[0] == lambda_prior_sample.shape[0]

    alpha_sample = []
    for alpha_val, lambda_val in zip(alpha_prior_sample, lambda_prior_sample):
        if lambda_val == main_peak:
            alpha_val_sampled = np.random.normal(loc=alpha_val, scale=0.05, size=1)
        elif lambda_val == 0.1:
            alpha_val_sampled = mixing_param * np.random.laplace(loc=alpha_val, scale=5e-4, size=1) + \
                                (1 - mixing_param) * np.random.laplace(loc=alpha_val, scale=0.05, size=1)
        else:
            raise ValueError('lambda_prior_sample needs to be either 0.9 or 0.1. Currently %s.' % lambda_val)

        alpha_sample.append(alpha_val_sampled)

    alpha_sample = np.clip(a=np.array(alpha_sample), a_min=-math.pi, a_max=math.pi)
    lambda_sample = np.array(lambda_prior_sample)

    return alpha_sample, lambda_sample


def generate_synthetic_galaxy(alpha_val, lambda_val, downsampling=0):
    # Setup Galsim values first
    gal_flux = 1e5  # counts
    gal_r0 = 2.7  # arcsec
    psf_beta = 5  #
    psf_re = 1.0  # arcsec
    pixel_scale = 0.3  # arcsec / pixel
    sky_level = 2.5e3  # counts / arcsec^2
    random_seed = 152332
    unit = coord.AngleUnit(1.0)

    # Initialize the (pseudo-)random number generator that we will be using below.
    # For a technical reason that will be explained later (demo9.py), we add 1 to the
    # given random seed here.
    rng = galsim.BaseDeviate(random_seed + 1)

    # Define the galaxy profile.
    gal = galsim.Exponential(flux=gal_flux, scale_radius=gal_r0)

    # Shear the galaxy by some value.
    # There are quite a few ways you can use to specify a shape.
    # q, beta      Axis ratio and position angle: q = b/a, 0 < q < 1
    # e, beta      Ellipticity and position angle: |e| = (1-q^2)/(1+q^2)
    # g, beta      ("Reduced") Shear and position angle: |g| = (1-q)/(1+q)
    # eta, beta    Conformal shear and position angle: eta = ln(1/q)
    # e1,e2        Ellipticity components: e1 = e cos(2 beta), e2 = e sin(2 beta)
    # g1,g2        ("Reduced") shear components: g1 = g cos(2 beta), g2 = g sin(2 beta)
    # eta1,eta2    Conformal shear components: eta1 = eta cos(2 beta), eta2 = eta sin(2 beta)
    gal = gal.shear(q=lambda_val, beta=galsim.Angle(alpha_val, unit).wrap())

    # Define the PSF profile.
    psf = galsim.Moffat(beta=psf_beta, flux=1., half_light_radius=psf_re)

    # Final profile is the convolution of these.
    final = galsim.Convolve([gal, psf])

    # Draw the image with a particular pixel scale.
    image = final.drawImage(scale=pixel_scale)

    # Normalize the input between 0 and 1
    image_array = (image.array - np.min(image.array)) / (np.max(image.array) - np.min(image.array))

    # Downsample
    if downsampling > 0 and downsampling != image_array.shape[0]:
        if downsampling > image_array.shape[0]:
            raise ValueError('Images is too small. Currently %d, wanting to subsample to %d.' % (
                image_array.shape[0], downsampling))

        # Figure out how much we need to fill it
        ch, cw = image_array.shape
        mult_factor = (ch // downsampling) + 1
        dh, dw = downsampling * mult_factor, downsampling * mult_factor

        # Create a black figure of the right size
        image_full = np.zeros((dh, dw))

        # compute center offset
        xx = (dw - cw) // 2
        yy = (dh - ch) // 2

        # copy img image into center of result image
        image_full[yy:yy + ch, xx:xx + cw] = image_array

        # downsample
        image_array = downsample(myarr=image_full, factor=mult_factor)

        # Normalize the input between 0 and 1
        image_array = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array))

    return image_array


def main(sample_size, main_peak, save_out=True, mixing_param=0.5, downsampling=20):
    # Sample from the prior first
    alpha_prior_sample, lambda_prior_sample = sample_from_prior(
        sample_size=sample_size, main_peak=main_peak)

    # Sample to get the true values
    alpha_sample, lambda_sample = sample_true_values(alpha_prior_sample=alpha_prior_sample,
                                                     lambda_prior_sample=lambda_prior_sample,
                                                     mixing_param=mixing_param, main_peak=main_peak)

    # Generate images
    param_mat = np.hstack((alpha_sample.reshape(-1, 1), lambda_sample.reshape(-1, 1)))
    final_sample = []
    pbar = tqdm(total=sample_size, desc='Simulating %d Galaxies.' % sample_size)
    for alpha_val, lambda_val in param_mat:
        try:
            galaxy_sample = generate_synthetic_galaxy(
                alpha_val=alpha_val, lambda_val=lambda_val, downsampling=downsampling)
            final_sample.append(galaxy_sample)
            pbar.update(1)
        except:
            pbar.update(1)
            continue

    final_sample = np.array(final_sample)
    if save_out:
        outfile_name = 'data/galsim_simulated_%sgals_%smainpeak_downsampling%s_%smixingparam_%s.pkl' % (
            final_sample.shape[0], str(main_peak).replace('.', '-'), downsampling, mixing_param,
            datetime.strftime(datetime.today(), '%Y-%m-%d-%H-%M')
        )
        pickle.dump(obj={'prior_mat': np.hstack((alpha_prior_sample.reshape(-1, 1),
                                                 lambda_prior_sample.reshape(-1, 1))),
                         'param_mat': param_mat,
                         'galaxies_generated': final_sample,
                         'downsampling': downsampling,
                         'main_peak': main_peak},
                    file=open(outfile_name, 'wb'), protocol=3)
    else:
        return final_sample


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_size', action="store", type=int, default=10000,
                        help='Simulated sample size.')
    parser.add_argument('--downsampling', action="store", type=int, default=20,
                        help='Shape of the downsampled image.')
    parser.add_argument('--main_peak', action="store", type=float, default=0.9,
                        help='Value of the main peak for the galaxies axis ratio.')
    argument_parsed = parser.parse_args()

    main(
        sample_size=argument_parsed.sample_size,
        downsampling=argument_parsed.downsampling,
        main_peak=argument_parsed.main_peak
    )
