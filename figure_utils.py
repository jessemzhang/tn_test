from __future__ import print_function

import time
import numpy as np
import pandas as pd
import seaborn as sns
import truncated_normal as tn
import matplotlib.pyplot as plt

from scipy.stats import norm, t, truncnorm, ttest_ind
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from statsmodels.distributions.empirical_distribution import ECDF


# ------------------------------------------------------------------------------
# For simulation
# ------------------------------------------------------------------------------

def samp_diag_norm(mu, Cov, n=1):
    samp = np.array([[
        np.random.normal(mu[i], np.sqrt(Cov[i])) for i in range(len(mu))
    ] for i in range(n)])
    return samp


def truncnormmv(mu, Cov, a, m):
    """Sample from multivariate truncated normal (MC approach)
       (diagonal covariance case)
    """
    samps = np.zeros((m, len(a)))
    i = 0
    while i < m:
        samp = samp_diag_norm(mu, Cov)
        if np.dot(samp, a) > 0:
            samps[i] = samp[0]
            i += 1
    return samps


def simulate_hyperplane(muL, muR, Cov, nY, nZ, zero_mean=False, method='linear'):
    """
    Finds a hyperplane by sampling from a multivariate normal distribution 
    based on muL, muR (diagonal covariance)
    """

    # Generate training set
    y = samp_diag_norm(muL, Cov, n=nY)
    z = samp_diag_norm(muR, Cov, n=nZ)
    data = np.vstack((y, z))
    labels = np.hstack((np.array([-1 for i in range(nY)]),
                        np.array([1 for i in range(nZ)])))
    
    # Zero-mean features (since we're not fitting an intercept)
    if zero_mean:
        mean = np.mean(data, 0)
        data -= mean
    
    # Get a separating hyperplane
    if method == 'linear':
        lr = LinearRegression(fit_intercept=False)
        lr.fit(data, labels)
        a = lr.coef_.reshape(-1)
        
    else:
        svm = SVC(kernel='linear', C=100.)
        svm.fit(data, labels)
        a = svm.coef_.reshape(-1)

    return y, z, a


def run_simulation(muL, muR, Cov, nY=49, nZ=55, num_sims=100, method='linear',
                   a_init=None, split_prop=0.5, zero_mean=True, eps=1e-3, verbose=False):
    
    k = len(muL)
    curves = {
        'T test': np.zeros((num_sims, k)),
        'TN test (var known, a given)': np.zeros((num_sims, k)),
        'TN test (var unknown, a given)': np.zeros((num_sims, k)),
        'TN test (var known, a estimated)': np.zeros((num_sims, k)),
        'TN test (var unknown, a estimated)': np.zeros((num_sims, k))
    }
    
    start = time.time()
    for i in range(num_sims):
        
        if a_init is None:
            _, _, a = simulate_hyperplane(muL, muR, Cov, nY, nZ)
        else:
            a = a_init

        y = truncnormmv(muL, Cov, -a, nY)
        z = truncnormmv(muR, Cov, a, nZ)

        curves['T test'][i] = ttest_ind(y, z)[1]
        curves['TN test (var known, a given)'][i] = tn.tn_test(y, z, a=a, var=Cov, 
                                                               eps=eps, method=method,
                                                               verbose=verbose)
        curves['TN test (var unknown, a given)'][i] = tn.tn_test(y, z, a=a, 
                                                                 eps=eps, method=method,
                                                                 verbose=verbose)
        curves['TN test (var known, a estimated)'][i] = tn.tn_test(y, z, var=Cov, 
                                                                   split_prop=split_prop, 
                                                                   zero_mean=zero_mean,
                                                                   eps=eps, method=method,
                                                                   verbose=verbose)
        curves['TN test (var unknown, a estimated)'][i] = tn.tn_test(y, z, 
                                                                     split_prop=split_prop,
                                                                     zero_mean=zero_mean,
                                                                     eps=eps, method=method,
                                                                     verbose=verbose)
        
        print('\r%s/%s simulations done (%.2f s elapsed).'\
              %(i+1, num_sims, time.time()-start), end='')
        
    return curves


def run_simulation_1D(a, muL, muR, var, nY=49, nZ=55, num_sims=100):

    curves = {
        'T test': np.zeros(num_sims),
        'TN test (var known)': np.zeros(num_sims),
        'TN test (var unknown)': np.zeros(num_sims)
    }
    
    start = time.time()
    for i in range(num_sims):

        y = truncnorm.rvs(a=-np.inf, b=a-muL, loc=muL, size=nY)
        z = truncnorm.rvs(a=a-muR, b=np.inf, loc=muR, size=nZ)

        # T test p value
        curves['T test'][i] = ttest_ind(y, z)[1]
        
        # Corrected p values (var known)
        eta1, eta2, eta3, loss = tn.get_natural_params_1D(y, z, a, var=var, num_iters=10000, 
                                                          learning_rate=10., verbose=False)
        var_hat = 1./eta1
        muL_hat = eta2/eta1
        muR_hat = eta3/eta1
        curves['TN test (var known)'][i] = tn.get_p_val_1D(y, z, a, muL_hat, muR_hat, 1)

        # Corrected p value (var unknown)
        eta1, eta2, eta3, loss = tn.get_natural_params_1D(y, z, a, num_iters=10000, 
                                                          learning_rate=10., verbose=False)
        var_hat = 1./eta1
        muL_hat = eta2/eta1
        muR_hat = eta3/eta1
        curves['TN test (var unknown)'][i] = tn.get_p_val_1D(y, z, a, muL_hat, muR_hat, var_hat)
        
        print('\r%s/%s simulations done (%.2f s elapsed).'\
              %(i+1, num_sims, time.time()-start), end='')
        
    return curves


# ------------------------------------------------------------------------------
# For visualization
# ------------------------------------------------------------------------------

def plot_labels_legend(x1, x2, Y, overlay=False, legend=True):
    if overlay:
        for i, label in enumerate(np.unique(Y)):
            plt.scatter(x1[Y == i], x2[Y == i], label=i, alpha=0.5, s=10)
            plt.annotate(label, 
                         [np.mean(x1[Y == i]), np.mean(x2[Y == i])],
                         horizontalalignment='center',
                         verticalalignment='center',
                         size=20, weight='bold', color='k') 
    else:
        for i in np.unique(Y):
            plt.plot(x1[Y == i], x2[Y == i], '.', label=i)
    if legend: plt.legend()


def plot_stacked_hist(v0, v1, title=None):
    """Plot two histograms on top of one another"""
    bins = np.histogram(np.hstack((v0, v1)), bins=20)[1]
    data = [v0, v1]
    plt.hist(data, bins, label=['0','1'], alpha=0.8, color=['r','g'],
             density=True, edgecolor='none')
    if title is not None: plt.title(title)
        

def pca_visualization(y, z):
    """Visualization via PCA of two groups"""
    pca = PCA(n_components=2)
    embed = pca.fit_transform(np.vstack((y, z)))
    plt.scatter(embed[:len(y), 0], embed[:len(y), 1], label='y')
    plt.scatter(embed[len(y):, 0], embed[len(y):, 1], label='z')
    plt.xlabel('pc1')
    plt.ylabel('pc2')
    plt.legend()
    plt.show()

    
def plot_hists(data, title=None, value_name=None, legend=True, add_markers=True,
               bins=10):
    """Plots multiple histograms, one for each key: value pair in data 
       (key = legend label, value = array of values to make hist of)
    """
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    markers = ['s', 'D', 'v', 'o', '*', '^', '+', '.']
#    df = pd.DataFrame.from_dict(data)
#    df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in data.items() ]))
    for i, label in enumerate(data):
        df = pd.DataFrame.from_dict({label: data[label]})
        ax = sns.distplot(df[label], rug=False, bins=bins)
        if add_markers:
            x = ax.lines[2*i].get_xdata()
            y = ax.lines[2*i].get_ydata()
            plt.plot(x[np.argmax(y)], np.max(y), marker=markers[i], markeredgecolor='k', 
                     color=colors[i], label=label)
    plt.xlabel('data' if value_name is None else value_name)
    if title is not None:
        plt.title(title)
    if legend: plt.legend()   
    

def plot_ecdf(data, label=None):
    ecdf = ECDF(data)
    x = np.linspace(min(data), max(data), 1000)
    y = ecdf(x)
    plt.plot(x, y, label=label)
    
    
def plot_ecdfs(curves, title=None, logmode=False, xlim=None, legend=True):
    markers = ['s', 'o', 'v', 'D', '+', '.', '^', '*']
    curves['Uniform'] = np.linspace(0.0001, 1, 1000)
    for i, label in enumerate(curves):
        plot_ecdf(-np.log10(curves[label]).reshape(-1), label=label)
        plt.xlabel('-log($p$)')
    if legend: plt.legend()
    if title is not None: plt.title(title)
    if logmode: plt.xscale('log')
    if xlim is not None: plt.xlim(xlim)
    
    
def plot_1D(y, z, a):
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plot_hists({'': np.hstack((y, z))}, value_name='gene expression',
               legend=False, add_markers=False, bins=20)
    plt.yticks([])
    plt.subplot(1, 2, 2)
    plt.yticks([])
    plot_hists({'c1': y, 'c2': z}, value_name='gene expression')
    plt.axvline(x=a, color='k', linestyle='--')
    plt.tight_layout()

    
def plot_2D(a, y, z, ylim=(-3, 3), muL=None, muR=None, s=None, legend=True):
    a_orth = np.array(np.flipud(a))
    a_orth[0] *= -1
    plt.scatter(y[:, 0], y[:, 1], label='y', s=s)
    plt.scatter(z[:, 0], z[:, 1], label='z', s=s)
    tx = np.linspace(np.min((np.min(y[:, 0]), np.min(z[:, 0]))),
                     np.max((np.max(y[:, 0]), np.max(z[:, 0]))), 3)
    plt.plot(tx, a_orth[1]/a_orth[0]*tx, '--', c='k', label='a')
    if muL is not None:
        plt.scatter(muL[0], muL[1], s=100, c='k', edgecolors='w')
    if muR is not None:
        plt.scatter(muR[0], muR[1], s=100, c='k', edgecolors='w')
    plt.xlabel('gene 1')
    plt.ylabel('gene 2')
    plt.ylim(ylim)
    plt.grid()
    if legend: plt.legend()

