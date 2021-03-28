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
import pandas as pd

from Estimator.stein import SteinScoreEstimator
from Estimator.spectral import SpectralScoreEstimator

#distr paramters
q_mean = 0.
q_logstd = 0.
q_std = np.exp(q_logstd)
q_precision = 1. / q_std ** 2


def main():

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
    simul_length = 150
    simul=range(simul_length)
    #vectors of length as loop
    collect_mse_stein = np.zeros(simul_length) 
    collect_mse_spectral = np.zeros(simul_length)
    
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

        stein_dlog_q_samples = stein.compute_gradients(samples[..., None])
        # stein_dlog_q_samples
        #squeeze removes dimesions size 1 from tensor
        stein_dlog_q_samples = tf.squeeze(stein_dlog_q_samples, -1)

        #define function Stein gradient calulation, for the Stein+
        #out of sample part of the estimation
        def stein_dlog(y):
            stein_dlog_qx = stein.compute_gradients(samples[..., None],
                                                x=y[..., None, None])
            # stein_dlog_qx: []
            stein_dlog_qx = tf.squeeze(stein_dlog_qx, axis=(-1, -2))
            return stein_dlog_qx

        # actualy values for stein+ stein_dlog_qx
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
        
    
            stein_true_mse =  true_dlog_qxs.size**(-1)*np.sum((stein_dlog_qxs-true_dlog_qxs)**2)
            spectral_true_mse = true_dlog_qxs.size**(-1)*np.sum((spectral_dlog_qxs-true_dlog_qxs)**2)
            collect_mse_stein[i-1] = stein_true_mse
            collect_mse_spectral[i-1] = spectral_true_mse
            
            
            
    #print info about average mse in several experiments        
    avrg_stein_mse = simul_length**(-1)*np.sum(collect_mse_stein)
    avrg_spectral_mse = simul_length**(-1)*np.sum(collect_mse_spectral)
    print('Average for Stein+ is {}'.format(avrg_stein_mse))
    print('Average for Spectral is {}'.format(avrg_spectral_mse))    
   
    

    #build pandas Dataframe with important data
    tableCompare = pd.DataFrame(columns=['Stein$^+$','Spectral'],
                                index=['Average MSE','min MSE','max MSE'])
    tableCompare.loc['Average MSE']=pd.Series({'Stein$^+$':round(avrg_stein_mse,4),
                                               'Spectral':round(avrg_spectral_mse,4)})
    tableCompare.loc['min MSE']=pd.Series({'Stein$^+$':round(np.amin(collect_mse_stein),4),
                                               'Spectral':round(np.amin(collect_mse_spectral),4)})
    tableCompare.loc['max MSE']=pd.Series({'Stein$^+$':round(np.amax(collect_mse_stein),4),
                                               'Spectral':round(np.amax(collect_mse_spectral),4)})
    print(tableCompare)
    
    
    #Saving the table as a plot
    from pandas.plotting import table  

    plt.figure(dpi=200)
    ax = plt.subplot(111, frame_on=False) # no visible frame
    ax.xaxis.set_visible(False)  # hide the x axis
    ax.yaxis.set_visible(False)  # hide the y axis

    table(ax, tableCompare)  
    plt.show()
    
    #histogram for distribution of MSE
    #Cut off High values to still show in histo
    which_val_high = collect_mse_spectral > 35
    collect_mse_spectral[which_val_high]=5
    which_val_high2 = collect_mse_spectral > 5
    collect_mse_spectral[which_val_high2]=4.87
    #plot histo
    bins = np.linspace(0, 5, 40)
    plt.figure(dpi=200)
    plt.title('MSE Comparison in {} Experiments, lambda={}'.format(simul_length,eta))
    plt.hist(collect_mse_stein, bins, alpha=0.5,
             label='$Stein^+$', color='skyblue')
    plt.hist(collect_mse_spectral, bins, alpha=0.5,
             label='Spectral', color='green')
    plt.legend(loc='upper right')
    plt.show()
    
    

if __name__ == "__main__":
    main()