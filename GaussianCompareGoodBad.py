# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 16:56:59 2020

@author: Acer
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import zhusuan as zs
import numpy as np


from Estimator.stein import SteinScoreEstimator
from Estimator.spectral import SpectralScoreEstimator


#distr parameters
q_mean = 0.
q_logstd = 0.
q_std = np.exp(q_logstd)
q_precision = 1. / q_std ** 2


def main():
    tf.set_random_seed(1234)
    np.random.seed(1234)

    from matplotlib import rc
    from matplotlib import pyplot as plt
    rc('text', usetex=True)


    #substantially better performance for lower eta for spectral, eg 0.1
    eta = 0.1
    n_eigen = 6
    #number of points
    n = 100
    #borders of plot
    lower_box = -3
    upper_box = 3
    #simulation number
    simul_length = 20
    simul = range(simul_length)
    #vectors of length as loop
    collect_mse_stein = np.zeros(simul_length) 
    collect_mse_spectral = np.zeros(simul_length)
    
    #plotting best and worst performaance
    minmsespec = 500
    maxmsespec = 0
    minspectral_dlog_qxs = 0
    minspec_stein_dlog_qxs = 0
    maxspectral_dlog_qxs = 0
    maxspec_stein_dlog_qxs = 0
    
    
    for i in simul:
        #creating true distr object
        q = zs.distributions.Normal(q_mean, logstd=q_logstd)
        # create samples samples
        samples = q.sample(n_samples=n)
        # variable definiton x
        x = tf.placeholder(tf.float32, shape=[None])
        # log_qx
        log_qx = q.log_prob(x)
        # true gradient of log prob true_dlog_qx
        true_dlog_qx = tf.map_fn(lambda i: tf.gradients(q.log_prob(i), i)[0], x)

        #Stein Gradient Estimator
        stein = SteinScoreEstimator(eta=eta)
        #see stein.py for further details
        stein_dlog_q_samples = stein.compute_gradients(samples[..., None])
        # stein_dlog_q_samples
        #squeeze removes dimesions size 1 from tensor
        stein_dlog_q_samples = tf.squeeze(stein_dlog_q_samples, -1)

        #define function Stein gradient calulation, for the Stein+
        #out of sample part of the estimation
        def stein_dlog(y):
            stein_dlog_qx = stein.compute_gradients(samples[..., None],
                                                x=y[..., None, None])
            # stein_dlog_qx
            stein_dlog_qx = tf.squeeze(stein_dlog_qx, axis=(-1, -2))
            return stein_dlog_qx

        # actualy values for stein+ 
        stein_dlog_qx = tf.map_fn(stein_dlog, x)

        #spectral series object + estimation
        spectral = SpectralScoreEstimator(n_eigen=n_eigen)
        spectral_dlog_qx = spectral.compute_gradients(samples[..., None],
                                                  x=x[..., None])
        # spectral_dlog_qx
        spectral_dlog_qx = tf.squeeze(spectral_dlog_qx, -1)
    


        #start the actual tensorflow process 
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            xs = np.linspace(lower_box, upper_box, n)
            log_qxs, true_dlog_qxs, spectral_dlog_qxs, stein_dlog_qxs, \
                stein_dlog_basis, samples_  = \
                    sess.run([log_qx, true_dlog_qx, spectral_dlog_qx, stein_dlog_qx,
                          stein_dlog_q_samples, samples],
                        feed_dict={x: xs})
        
            #mse
            stein_true_mse =  true_dlog_qxs.size**(-1)*np.sum((stein_dlog_qxs-true_dlog_qxs)**2)
            spectral_true_mse = true_dlog_qxs.size**(-1)*np.sum((spectral_dlog_qxs-true_dlog_qxs)**2)
            collect_mse_stein[i-1] = stein_true_mse
            collect_mse_spectral[i-1] = spectral_true_mse
            #collect good and bad approx
            if spectral_true_mse > maxmsespec:
                maxmsespec = spectral_true_mse
                maxspectral_dlog_qxs = spectral_dlog_qxs
                maxspec_stein_dlog_qxs = stein_dlog_qxs
            elif spectral_true_mse < minmsespec:
                minmsespec = spectral_true_mse
                minspectral_dlog_qxs = spectral_dlog_qxs
                minspec_stein_dlog_qxs = stein_dlog_qxs
                
                          
    
    #plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, dpi=200)
    fig.suptitle('Good and Bad Estiamtion Examples')
    ax1.set(ylabel='MSE={}'.format(round(minmsespec,4)))
    ax2.set(ylabel='MSE={}'.format(round(maxmsespec,4)))
    ax1.plot(xs, true_dlog_qxs,"--", label=r"$\nabla_x\log q(x)$", linewidth=2.)
    ax2.plot(xs, true_dlog_qxs,"--", label=r"$\nabla_x\log q(x)$", linewidth=2.)

    ax1.plot(xs,  minspectral_dlog_qxs,
                 label=r"$\hat{\nabla}_x\log q(x)$, Spectral", linewidth=2.)
    ax1.plot(xs,minspec_stein_dlog_qxs, 
                 label=r"$\hat{\nabla}_x\log q(x)$, Stein$^+$", linewidth=2.)
    ax2.plot(xs,  maxspectral_dlog_qxs,
                 label=r"$\hat{\nabla}_x\log q(x)$, Spectral", linewidth=2.)
    ax2.plot(xs,maxspec_stein_dlog_qxs, 
                 label=r"$\hat{\nabla}_x\log q(x)$, Stein$^+$", linewidth=2.)
    ax1.legend(loc='best', fontsize='x-small')
if __name__ == "__main__":
    main()