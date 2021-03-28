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

#distr paramteres
q_mean = 0.
q_logstd = 0.
q_std = np.exp(q_logstd)
q_precision = 1. / q_std ** 2


#true impl relationship parameters
psi_true=3
mu=np.array([1,3*1])
C=np.array([[1,3],[3,9.1]])
from numpy.linalg import inv
C_inv=inv(C)
det_C=0.1
sigma_Y2=np.sqrt(0.1)



def main():
    #functions
    def f_true(x):
        result=psi_true*x+np.random.normal(0,sigma_Y2)
        return result

    def f_phi(x, eps,phi):
        result = phi*x+eps
        return result

    #differentation of stein and spectral parameters
    phist=-1
    phisp=-1
    delta_phist_f = phist
    delta_phisp_f = phisp

    #joint dens
    def px_zjoint(x,z):
        xz=np.array([x,z])-mu
        result=(2*3.1415)**(-1)*det_C**(-0.5)*np.exp(-0.5*
                                np.matmul(xz,np.matmul(C_inv, xz)))
        return result
    
    #grad joint
    #now claculated generally for normal 2-dim distributions
    def grad_z_joint(x,z):
        grad=C_inv[1,0]*mu[0]+C_inv[1,1]*mu[1]-C_inv[0,1]*x-C_inv[1,1]*z
        return grad
    
    x_size=75
    x_sample=np.random.normal(loc=mu[0],size=x_size)
    numbapprox=75
    
    #Evidence lower bound
    def gradientELBO(steingrads, idx,phi,delta_phi):
        eps1=np.random.normal(0,sigma_Y2,size=numbapprox)
        part1=np.mean(grad_z_joint(x_sample, f_phi(x_sample,eps1,phi))*x_sample)
        part2=np.mean(steingrads*x_sample)
        return part1-part2


    from matplotlib import rc
    from matplotlib import pyplot as plt
    rc('text', usetex=True)


    #simulation number
    simul_length=30
    simul=range(simul_length)
    eta = 1
    learningrate=np.repeat(0.01,simul_length+1)
    n_eigen = 6
   


   #only true for sigma^2 of eps is 0.1
    q = zs.distributions.Normal(q_mean, std=0.316)
    
    error_collect_stein=np.zeros(simul_length)
    error_collect_spectral=np.zeros(simul_length)
    for i in simul:
        print(i)
        idx=np.random.randint(0,x_size)
        #samples represent the epsilon
        samples = q.sample(n_samples=numbapprox)
        samples_st = f_phi(x_sample, samples,phist)
        samples_sp = f_phi(x_sample, samples,phisp)

        # variable definiton x
        x = tf.placeholder(tf.float32, shape=[None])
        # log_qx
        log_qx = q.log_prob(x)

        #Stein Gradient Estimator
        stein = SteinScoreEstimator(eta=eta)
        #see stein.py for further info
        stein_dlog_q_samples = stein.compute_gradients(samples_st[..., None])
        # stein_dlog_q_samples
        #squeeze removes dimesions size 1 from tensor
        stein_dlog_q_samples = tf.squeeze(stein_dlog_q_samples, -1)

        #define function Stein gradient calulation, for the Stein+
        #out of sample part of the estimation
        def stein_dlog(y):
            stein_dlog_qx = stein.compute_gradients(samples_st[..., None],
                                                x=y[..., None, None])
            # stein_dlog_qx
            stein_dlog_qx = tf.squeeze(stein_dlog_qx, axis=(-1, -2))
            return stein_dlog_qx

        # actualy values for stein+ stein_dlog_qx
        stein_dlog_qx = tf.map_fn(stein_dlog, x)

        #spectral series object + estimation
        spectral = SpectralScoreEstimator(n_eigen=n_eigen)
        spectral_dlog_qx = spectral.compute_gradients(samples_sp[..., None],
                                                  x=x[..., None])
        # spectral_dlog_qx
        spectral_dlog_qx = tf.squeeze(spectral_dlog_qx, -1)
    

        #start the actual tensorflow process 
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            log_qxs, spectral_dlog_qxs, stein_dlog_qxs, \
                stein_dlog_basis, samples_  = \
                    sess.run([log_qx, spectral_dlog_qx, stein_dlog_qx,
                          stein_dlog_q_samples, samples],
            feed_dict={x: x_sample})
            
            
            #plus in grad descent since we maximise
            phist = phist + learningrate[i]*gradientELBO(
                stein_dlog_qxs, idx,phist,delta_phist_f)
            phisp = phisp + learningrate[i]*gradientELBO(
                spectral_dlog_qxs, idx,phisp,delta_phisp_f)
            delta_phist_f=phist
            delta_phisp_f=phisp
            #calc predictions & true val
            z_hat_stein=f_phi(x_sample, np.random.normal(0,sigma_Y2, x_sample.size),phist)
            z_hat_spectral=f_phi(x_sample, np.random.normal(0,sigma_Y2, x_sample.size),phisp)
            z_true=f_true(x_sample)
            
            #collect the error terms
            error_collect_stein[i]=((z_hat_stein-z_true)**2).mean()
            error_collect_spectral[i]=((z_hat_spectral-z_true)**2).mean()


    #Calc for plotting
    z_hat_stein=f_phi(x_sample, np.random.normal(0,sigma_Y2, x_sample.size),phist)
    z_hat_spectral=f_phi(x_sample, np.random.normal(0,sigma_Y2, x_sample.size),phisp)
    z_true=f_true(x_sample)
    
    #histogram of prediction
    bins = np.linspace(-5, 15, 30)
    plt.figure(dpi=200)
    plt.title('Comparison of True and $Stein^+$ Distribution')
    plt.hist(z_hat_stein, bins, alpha=0.5,
             label='Stein$^+$, $\phi={val}$'.format(val=round(phist,4)), color='skyblue', density=True)
    plt.hist(z_true, bins, alpha=0.5,
             label='true, $\psi={}$'.format(psi_true), color='green', density=True)
    #plt.plot(bins, norm.pdf(bins, loc=mu[0], scale=3))
    plt.legend(loc='upper right')
    plt.show()
    
    #histogram of prediction
    bins = np.linspace(-5, 15, 30)
    plt.figure(dpi=200)
    plt.title('Comparison of True and Spectral Distribution')
    plt.hist(z_hat_spectral, bins, alpha=0.5,
             label='$Spectral$, $\phi={val}$'.format(val=round(phisp,4)), color='orange', density=True)
    plt.hist(z_true, bins, alpha=0.5,
             label='true, $\psi={}$'.format(psi_true), color='green', density=True)
    plt.legend(loc='upper right')
    plt.show()
    
    #plotting error over time
    #finding values below treshhold
    tresh_st_1 = np.where(error_collect_stein<= 1)[0][0]
    tresh_st_01 = np.where(error_collect_stein<= 0.1)[0][0]
    tresh_sp_1 = np.where(error_collect_spectral<= 1)[0][0]
    tresh_sp_01 = np.where(error_collect_spectral<= 0.1)[0][0]
    
    tresh_st = np.array((tresh_st_1,tresh_st_01))
    tresh_sp = np.array((tresh_sp_1,tresh_sp_01))
    #actual plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, dpi=150)
    fig.suptitle('Comparison of Training Error')
    ax1.plot(range(0,simul_length),error_collect_stein,
             label='Stein$^+$', color='skyblue', markevery=tresh_st,
             marker='*')
    #markers for threshhold in plot
    ax1.annotate('$\leq 1$',(tresh_st_1,error_collect_stein[tresh_st_1])
                 ,bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=0.5),
                 xytext=(tresh_st_1+0.85,error_collect_stein[tresh_st_1]+1.5))
    ax1.annotate('$\leq 0.1$',(tresh_st_01,error_collect_stein[tresh_st_01])
                 ,bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=0.5),
                 xytext=(tresh_st_01+0.85,error_collect_stein[tresh_st_01]+1.5))
    
    ax2.plot(range(0,simul_length),error_collect_spectral,
             label='Spectral', color='orange', markevery=tresh_sp,
             marker='*')
    #markers for threshhold in plot
    ax2.annotate('$\leq 1$',(tresh_sp_1,error_collect_stein[tresh_sp_1])
                 ,bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=0.5),
                 xytext=(tresh_sp_1+0.85,error_collect_stein[tresh_sp_1]+1.5))
    ax2.annotate('$\leq 0.1$',(tresh_sp_01,error_collect_stein[tresh_sp_01])
                 ,bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=0.5),
                 xytext=(tresh_sp_01+0.85,error_collect_stein[tresh_sp_01]+1.5))
    ax1.legend(loc='best')
    ax2.legend(loc='best')
    plt.ylim(0,15)
        
if __name__ == "__main__":
    main()