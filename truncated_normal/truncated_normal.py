"""
Module for running the TN test as described in the paper "Towards a post-
clustering test for differential expression".

https://www.biorxiv.org/content/early/2018/11/05/463265

Written by Jesse Min Zhang 11/8/18
"""

from __future__ import print_function

import time
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy.stats import norm, t
from scipy.integrate import quad
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression

# ------------------------------------------------------------------------------
# Basic functions
# ------------------------------------------------------------------------------

def phi(z):
    """pdf for N(0, 1) distribution"""
    return norm.pdf(z)


def Phi(z):
    """cdf for N(0, 1) distribution"""
    return norm.cdf(z)


def trunccdf(x, a, b, mu, var=1):
    """CDF of a truncated normal distribution"""
    xi = (x-mu)/np.sqrt(var)
    alpha = (a-mu)/np.sqrt(var)
    beta = (b-mu)/np.sqrt(var)
    Z = Phi(beta)-Phi(alpha)
    return (Phi(xi)-Phi(alpha))/Z


def truncstats(a, b, mu, var=1):
    """Mean and variance of a truncated normal distribution
       (left truncation at a, right truncation at b)
    """
    alpha = (a-mu)/np.sqrt(var)
    beta = (b-mu)/np.sqrt(var)
    Z = Phi(beta)-Phi(alpha)
    if a == -np.inf:
        alpha_ = 0
    else:
        alpha_ = alpha*phi(alpha)
    if b == np.inf:
        beta_ = 0
    else:
        beta_ = beta*phi(beta)
    truncmean = mu + np.sqrt(var)*(phi(alpha)-phi(beta))/Z
    truncvar = var*(1+(alpha_-beta_)/Z - ((phi(alpha)-phi(beta))/Z)**2)
    return truncmean, truncvar


# ------------------------------------------------------------------------------
# Get parameters for X and W
# ------------------------------------------------------------------------------


def get_null_trunc_params(muL, muR, var=1, a=0):
    """Gets first and second order moments of pdf Y, pdf Z
       assuming theta=0 (i.e. both distributions have same mean)
    """
    mu = (muL+muR)/2
    muY, varY = truncstats(-np.inf, a, mu, var=var)
    muZ, varZ = truncstats(a, np.inf, mu, var=var)
    return muY, varY, muZ, varZ


def get_marginals(a, b, muL, muR, var=None, ind=0):
    """Gets the marginal distributions of a gene given a as the truncation plane

       var can either be None, in which case it's set to all ones, or
       a vector of estimated variances for each gene
       (diagonal covariance assumption)
    """

    assert len(a) == len(muL) == len(muR)

    if var is None:
        var = np.ones(len(a))

    # Separating coefficient of interest from other coefficients
    v_in, v_out = var[ind], np.delete(var, ind)
    a_in, a_out = a[ind], np.delete(a, ind)
    muL_in, muL_out = muL[ind], np.delete(muL, ind)
    muR_in, muR_out = muR[ind], np.delete(muR, ind)

    def pdfY(Y):
        C = 1/(Phi((-np.dot(a, muL)-b)/np.sqrt(np.sum(a**2*var)))*np.sqrt(2*np.pi*v_in))
        if len(a_out) == 0:
            PhiY = lambda x: 1 if x <= 0 else 0
        else:
            PhiY = lambda x: Phi((-a_in*x-np.dot(a_out, muL_out)-b)/np.sqrt(np.sum(a_out**2*v_out)))
        return C*np.exp(-(Y-muL_in)**2/(2*v_in))*PhiY(Y)

    def pdfZ(Z):
        C = 1/(Phi((np.dot(a, muR)+b)/np.sqrt(np.sum(a**2*var)))*np.sqrt(2*np.pi*v_in))
        if len(a_out) == 0:
            PhiZ = lambda x: 1 if x > 0 else 0
        else:
            PhiZ = lambda x: Phi((a_in*x+np.dot(a_out, muR_out)+b)/np.sqrt(np.sum(a_out**2*v_out)))
        return C*np.exp(-(Z-muR_in)**2/(2*v_in))*PhiZ(Z)

    return pdfY, pdfZ


def get_truncmv_params(a, b, muL, muR, var=None, ind=0):
    """Gets first and second moments via numerical integration"""
    pdfY, pdfZ = get_marginals(a, b, muL, muR, var=var, ind=ind)
    muY = quad(lambda x: x*pdfY(x), -np.inf, np.inf)[0]
    varY = quad(lambda x: x**2*pdfY(x), -np.inf, np.inf)[0]-muY**2
    muZ = quad(lambda x: x*pdfZ(x), -np.inf, np.inf)[0]
    varZ = quad(lambda x: x**2*pdfZ(x), -np.inf, np.inf)[0]-muZ**2
    return muY, varY, muZ, varZ


def get_null_truncmv_params(a, b, muL, muR, var=None, ind=0):
    """Gets first and second order moments of pdf Y, pdf Z
       assuming theta=0 (i.e. both distributions have same mean on index ind)
       (multivariate case)
    """
    mu = (muL[ind]+muR[ind])/2.
    muL_, muR_ = np.array(muL), np.array(muR)
    muL_[ind], muR_[ind] = mu, mu
    muY, varY, muZ, varZ = get_truncmv_params(a, b, muL_, muR_, var=var, ind=ind)
    return muY, varY, muZ, varZ


# ------------------------------------------------------------------------------
# Estimation of muL, muR using gradient ascent
# ------------------------------------------------------------------------------

def get_natural_params_diagonal(y, z, a,
                                b=0,
                                num_iters=1000,
                                learning_rate=0.1,
                                eps=1e-5,
                                verbose=False,
                                opt_method='adagrad',
                                num_stability=1e-10,
                                var=None):
    """
    ML estimator for the natural parameters of the joint truncated Gaussian model
    Diagonal covariance matrix assumed.

    Notes on optimization:
       - Only make updates to entries of eta1 that result in positive values

    Parameters
    ----------
    y: points from one cluster
    z: points from the other cluster
    a: separating hyperplane
    b: intercept term
    num_iters: max number of optimization steps
    learning_rate: learning rate for each parameter equals this value times the
        inverse of variance in corresponding entry
    eps: if the likelihood changes by less than this amount, terminate early
    verbose: whether or not to print progress
    opt_method: either 'adagrad', 'rmsprop', or None
    num_stability: for the denominator when using adagrad or rmsprop
    var: if variance is given, do not update eta1

    Returns
    ----------
    eta1: first natural parameter
    eta2: second natural parameter
    eta3: third natural parameter
    likelihood: vector of log likelihood values during training
    """

    k = len(a)
    nY = len(y)
    nZ = len(z)

    # initialize parameters
    eta1 = 1./var if var is not None else np.random.uniform(0.5, 2, k)
    eta2 = np.random.randn(k)
    eta3 = np.random.randn(k)

    # first and second moments from data
    y1 = np.sum(y, 0)
    z1 = np.sum(z, 0)
    y2 = np.sum(y*y, 0)
    z2 = np.sum(z*z, 0)

    def log_likelihood(eta1, eta2, eta3):
        cY = (-np.dot(a, eta2/eta1)-b)/np.sqrt(np.dot(a, a/eta1))
        cZ = (np.dot(a, eta3/eta1)+b)/np.sqrt(np.dot(a, a/eta1))
        psi = nY*np.dot(eta2, eta2/eta1)/2 + nZ*np.dot(eta3, eta3/eta1)/2 + \
            nY*np.log(Phi(cY)) + nZ*np.log(Phi(cZ)) + (nY+nZ)*np.sum(np.log(1/eta1))/2
        return -0.5*np.dot(eta1, y2+z2) + np.dot(eta2, y1) + np.dot(eta3, z1) - psi

    # Learning rate
    learning_rate_ = learning_rate/np.var(np.vstack((y, z)), 0)

    # Gradient cache
    cache = {1: np.zeros(k), 2: np.zeros(k), 3: np.zeros(k)}

    if verbose:
        start = time.time()

    likelihood = [log_likelihood(eta1, eta2, eta3)]

    for step in range(num_iters):
        ieta1 = eta1**-1
        sigma = np.sqrt(np.dot(a, ieta1*a))
        cY = (-np.dot(a, ieta1*eta2)-b)/sigma
        cZ = (np.dot(a, ieta1*eta3)+b)/sigma

        # gradient updates
        deta2 = y1 - nY*ieta1*eta2 + nY*phi(cY)*ieta1*a/(sigma*Phi(cY)+num_stability)
        deta3 = z1 - nZ*ieta1*eta3 - nZ*phi(cZ)*ieta1*a/(sigma*Phi(cZ)+num_stability)

        if var is None:
            dphi1 = nY*eta2*eta2 + nZ*eta3*eta3 + (nY+nZ)*eta1 \
                    - nY*(phi(cY)/(Phi(cY)+num_stability))* \
                      (2*a*eta2/sigma - (np.dot(a, ieta1*eta2)+b)/sigma**3*a**2) \
                    - nZ*(phi(cZ)/(Phi(cZ)+num_stability))* \
                      ((np.dot(a, ieta1*eta3)+b)/sigma**3*a**2 - 2*a*eta3/sigma)
            deta1 = -y2 - z2 + ieta1*dphi1*ieta1
            deta1 /= 2.

        # Adagrad
        if opt_method == 'adagrad':
            if var is None:
                cache[1] += deta1**2
            cache[2] += deta2**2
            cache[3] += deta3**2

        # RMSprop
        elif opt_method == 'RMSprop':
            decay_rate = 0.99
            if var is None:
                cache[1] = decay_rate * cache[1] + (1 - decay_rate) * deta1**2
            cache[2] = decay_rate * cache[2] + (1 - decay_rate) * deta2**2
            cache[3] = decay_rate * cache[3] + (1 - decay_rate) * deta3**2

        else:
            cache[1] = np.ones(k)
            cache[2] = np.ones(k)
            cache[3] = np.ones(k)

        # update estimates
        if var is None:
            update = learning_rate_*deta1/(np.sqrt(cache[1])+num_stability)
            update_inds = eta1 > -update
            eta1[update_inds] += update[update_inds]
        eta2 += learning_rate_*deta2/(np.sqrt(cache[2])+num_stability)
        eta3 += learning_rate_*deta3/(np.sqrt(cache[3])+num_stability)

        likelihood.append(log_likelihood(eta1, eta2, eta3))

        # check convergence
        if np.abs(likelihood[-1]-likelihood[-2]) < eps:
            break

        if verbose:
            print('\r%s iterations done; likelihood = %.2e (%.2f s elapsed).'\
                  %(step+1, likelihood[-1], time.time()-start), end='')

    if verbose:
        print('')

    return eta1, eta2, eta3, likelihood


def get_natural_params_1D(y, z, a,
                          num_iters=1000,
                          learning_rate=0.1,
                          eps=1e-5,
                          opt_method='adagrad',
                          num_stability=1e-10,
                          verbose=True,
                          var=None):
    """
    ML estimator for the natural parameters of the joint truncated Gaussian model
    1D case

    Parameters
    ----------
    y: points from one cluster
    z: points from the other cluster
    a: separating hyperplane (threshold scalar)
    num_iters: max number of optimization steps
    learning_rate: learning rate for each parameter equals this value times the
        inverse of variance in corresponding entry
    eps: if the likelihood changes by less than this amount, terminate early
    verbose: whether or not to print progress
    opt_method: either 'adagrad', 'rmsprop', or None
    num_stability: for the denominator when using adagrad or rmsprop
    var: if variance is given, do not update eta1

    Returns
    ----------
    eta1: first natural parameter
    eta2: second natural parameter
    eta3: third natural parameter
    likelihood: vector of log likelihood values during training
    """

    assert isinstance(a, (int, float))
    assert isinstance(y[0], (int, float))
    assert isinstance(z[0], (int, float))

    nY = len(y)
    nZ = len(z)
    learning_rate /= np.max((nY, nZ))

    # first and second moments from data
    y1 = np.sum(y)
    z1 = np.sum(z)
    y2 = np.sum(y**2)
    z2 = np.sum(z**2)

    def log_likelihood(eta1, eta2, eta3):
        cY = a*np.sqrt(eta1)-eta2/np.sqrt(eta1)
        cZ = eta3/np.sqrt(eta1)-a*np.sqrt(eta1)
        psi = nY*eta2**2/(2*eta1) + nZ*eta3**2/(2*eta1) + nY*np.log(Phi(cY)) + \
              nZ*np.log(Phi(cZ))-(nY+nZ)*np.log(eta1)/2
        return -0.5*eta1*(y2+z2) + eta2*y1 + eta3*z1 - psi

    start = time.time()

    while True:

        # initialize parameters
        eta1 = 1./var if var is not None else np.random.uniform(0.5, 1)
        eta2 = np.random.randn()
        eta3 = np.random.randn()

        likelihood = [log_likelihood(eta1, eta2, eta3)]

        # Gradient cache
        cache = np.zeros(3)

        for step in range(num_iters):
            cY = a*np.sqrt(eta1)-eta2/np.sqrt(eta1)
            cZ = eta3/np.sqrt(eta1)-a*np.sqrt(eta1)

            # gradient updates
            deta2 = y1 - nY*eta2/eta1 + (1/np.sqrt(eta1))*nY*phi(cY)/Phi(cY)
            deta3 = z1 - nZ*eta3/eta1 - (1/np.sqrt(eta1))*nZ*phi(cZ)/Phi(cZ)

            if var is None:
                deta1 = -(y2+z2) \
                        + nY*eta2**2/eta1**2 \
                        + nZ*eta3**2/eta1**2 \
                        + (nZ+nY)/eta1 \
                        - nY*phi(cY)/Phi(cY)*(a*eta1**-0.5 + eta2*eta1**-1.5) \
                        + nZ*phi(cZ)/Phi(cZ)*(a*eta1**-0.5 + eta3*eta1**-1.5)
                deta1 /= 2.

            # Adagrad
            if opt_method == 'adagrad':
                if var is None:
                    cache[0] += deta1**2
                cache[1] += deta2**2
                cache[2] += deta3**2

            # RMSprop
            elif opt_method == 'RMSprop':
                decay_rate = 0.99
                if var is None:
                    cache[0] = decay_rate * cache[0] + (1 - decay_rate) * deta1**2
                cache[1] = decay_rate * cache[1] + (1 - decay_rate) * deta2**2
                cache[2] = decay_rate * cache[2] + (1 - decay_rate) * deta3**2

            # No adaptivity
            else:
                cache = np.ones(3)

            # update estimates
            if var is None:
                eta1_ = eta1 + learning_rate*deta1/(np.sqrt(cache[0])+num_stability)
                if eta1_ > 0:
                    eta1 = eta1_ # ensure that eta1 stays positive
            eta2 += learning_rate*deta2/(np.sqrt(cache[1])+num_stability)
            eta3 += learning_rate*deta3/(np.sqrt(cache[2])+num_stability)

            likelihood.append(log_likelihood(eta1, eta2, eta3))

            # check convergence
            if np.abs(likelihood[-1]-likelihood[-2]) < eps or np.isnan(likelihood[-1]):
                break

            if verbose:
                print('\r%s/%s gradient udpates performed (%.2f s elapsed).'
                      %(step+1, num_iters, time.time()-start), end='')

        if not np.isnan(likelihood[-1]):
            break

    if verbose:
        print('')

    return eta1, eta2, eta3, likelihood


# ------------------------------------------------------------------------------
# Compute corrected p value
# ------------------------------------------------------------------------------

def get_p_val(y, z, a, b, muL, muR, var, ind=0, use_tdist=False):
    """
    Correct pval approach using approximations of truncated Gaussians

    Parameters
    ----------
    y: points from one cluster
    z: points from the other cluster
    a: separating hyperplane
    use_tdist: null distribution of TN statistic to use
        False for standard normal
        True for t distribution with df=len(y)+len(z)-2
    muL, muR, var: estimated using maximum likelihood
    ind: gene to test

    Returns
    ----------
    pvalue
    """
    muY, varY, muZ, varZ = get_null_truncmv_params(a, b, muL, muR, var=var, ind=ind)
    nY, nZ = len(y), len(z)
    stat = (np.sum(z[:, ind])-np.sum(y[:, ind])-(nZ*muZ-nY*muY))/np.sqrt(nY*varY+nZ*varZ)
    if use_tdist:
        df = len(z)+len(y)-2
        d0 = t(df=df).cdf
    else:
        d0 = norm.cdf
    p = np.min((d0(stat), d0(-stat)))*2
    return p


def get_p_val_1D(y, z, a, muL, muR, var, use_tdist=False):
    """
    Correct pval approach using approximations of truncated Gaussians
    (1D case, so y, z, a should be scalars)

    Parameters
    ----------
    y: points from one cluster
    z: points from the other cluster
    a: threshold
    use_tdist: null distribution of TN statistic to use
        False for standard normal
        True for t distribution with df=len(y)+len(z)-2
    muL, muR, var: estimated using maximum likelihood

    Returns
    ----------
    pvalue
    """
    muY, varY, muZ, varZ = get_null_trunc_params(muL, muR, var=var, a=a)
    nY, nZ = len(y), len(z)
    stat = (np.sum(z)-np.sum(y)-(nZ*muZ-nY*muY))/np.sqrt(nY*varY+nZ*varZ)
    if use_tdist:
        df = len(z)+len(y)-2
        d0 = t(df=df).cdf
    else:
        d0 = norm.cdf
    p = np.min((d0(stat), d0(-stat)))*2
    return p


# ------------------------------------------------------------------------------
# TN test
# ------------------------------------------------------------------------------

def split_and_est_a(y, z,
                    verbose=False,
                    split_prop=0.5,
                    C=100,
                    return_consistency=False):
    """
    Given samples from two clusters, merge them into one dataset, then split that
    dataset into dataset 1 and dataset 2. Use dataset 1 to estimate a separating
    hyperplane; then use the hyperplane to assign labels to the samples
    in dataset 2.

    Parameters
    ----------
    y: points from one cluster
    z: points from the other cluster
    split_prop: proportion of samples to put into dataset 1
    C: tolerance parameter for svm
    return_consistency: whether or not to return how consistent fitted 
        hyperplane labels are with original ones (for dataset2)

    Returns
    ----------
    y: dataset 2 samples from cluster 1 after relabeling
    z: dataset 2 samples from cluster 2 after relabeling
    a: hyperplane estimated from dataset 1
    """

    if verbose:
        start = time.time()

    data = np.vstack((y, z))
    labels = np.hstack((np.zeros(len(y)), np.ones(len(z))))
    
    # Split dataset in half
    inds1 = np.zeros(len(data)).astype(bool)
    samp = np.sort(np.random.choice(range(len(data)), int(len(data)*split_prop), replace=False))
    inds1[samp] = True
    data1 = data[inds1]
    labels1 = labels[inds1]
    data2 = data[~inds1]
    labels2true = labels[~inds1]

    # Use first half to fit hyperplane, assign labels to second half
    svm = SVC(kernel='linear', C=C)
    svm.fit(data1, labels1)
    a = svm.coef_.reshape(-1)
    b = svm.intercept_
    labels2 = svm.predict(data2)

    y = data2[labels2 == 0]
    z = data2[labels2 == 1]
    consistency = np.sum(labels2 == labels2true)/float(len(labels2))
    if verbose:
        print('Labels assigned.. (%.2f s elapsed)'%(time.time()-start))
        print('Consistency of new labels with old: %.3f'\
              %(consistency))

    if return_consistency:
        return y, z, a, b, consistency
    
    return y, z, a, b


def tn_test(y, z,
            a=None,
            b=0,
            C=100,
            genes_to_test=None,
            var=None,
            verbose=False,
            return_split=False,
            split_prop=0.5,
            num_iters=100000,
            learning_rate=0.1,
            eps=1e-5,
            use_tdist=False,
            plot_likelihood=False,
            return_likelihood=False,
            return_hyperplane=False,
            return_consistency=False,
            num_cores=1):
    """
    The TN test as described in
    https://www.biorxiv.org/content/early/2018/11/05/463265.

    This test assumes that y and z are generated by normal distributions that
    underwent truncation (via a hyperplane) due to clustering. We estimate the
    variance and mean parameters via maximum likelihood, then we test to see if
    the estimated mean parameters are the same.

    Parameters
    ----------
    y: points from one cluster
    z: points from the other cluster
    a: separating hyperplane. estimated via data splitting if not given
    b: an intercept for the linear separator
    C: tolerance parameter for svm
    genes_to_test: list of the entries to run the test for. If not given, just
        test each of them (genes_to_test = range(a))
    var: if variance is given, do not update eta1 during optimization
    verbose: whether or not to print progress of test
    return_split: whether or not to return the y's and z's after data splitting
        and relabeling (a should be None else the input y, z will be returned)
    split_prop: proportion of samples to put into dataset 1
    num_iters: max number of optimization steps for maximum likelihood
    learning_rate: learning rate for each parameter equals this value times the
        inverse of variance in corresponding entry
    eps: if the likelihood changes by less than this amount, terminate early
    use_tdist: null distribution of TN statistic to use
        False for standard normal
        True for t distribution with df=len(y)+len(z)-2
    plot_likelihood: if true, generates plot of likelihood during optimization
    return_likelihood: if true, return the likelihood value being tracked
    return_hyperplane: if true, return the estimated separating hyperplane
    return_consistency: whether or not to return how consistent fitted 
        hyperplane labels are with original ones (for dataset2)
    num_cores: number of workers to parallelize on

    Returns
    ----------
    pvalue
    y, z if return_split is set to True
    """

    if verbose:
        start = time.time()

    if a is None:
        print("""
        WARNING: we recommend providing hyperplane a after running the clustering 
        algorithm on only the first half of the dataset. If a is not provided, this
        function randomly splits the datasets (see split_and_est_a()). Going this 
        route technically is not correct since the points in the second half were 
        used to cluster the points in the first half.
        
        If the clustering assignments of the pre-split dataset are known (e.g. for 
        synthetic data), then this shouldn't be an issue.
        """)
        if return_consistency:
            y, z, a, b, consistency = split_and_est_a(y, z, C=C, verbose=verbose,
                                                      split_prop=split_prop,
                                                      return_consistency=True)
        else:
            y, z, a, b = split_and_est_a(y, z, C=C, verbose=verbose, split_prop=split_prop)

    if genes_to_test is None:
        genes_to_test = range(len(a))

    # Remove genes with 0 variance across cells
    remove_inds = np.var(np.vstack((y, z)), 0) == 0
    y_out = np.array(y)
    z_out = np.array(z)
    y = y[:, ~remove_inds]
    z = z[:, ~remove_inds]
    a = a[~remove_inds]
    
    if var is not None:
        var = var[~remove_inds]
    
    if verbose:
        print('Number of genes with 0 variance across cells: %s'\
              %(np.sum(remove_inds)))
        
    # Remap genes to test
    d = {}
    for i, remove in enumerate(remove_inds):
        if not remove:
            d[i] = len(d)
    genes_to_test_filt = [d[i] for i in genes_to_test if i in d]
    
    eta1, eta2, eta3, likelihood = get_natural_params_diagonal(y, z, a, b=b,
                                                               var=var,
                                                               num_iters=num_iters,
                                                               learning_rate=learning_rate,
                                                               verbose=verbose,
                                                               eps=eps)
    var_hat = eta1**-1
    muL_hat = var_hat*eta2
    muR_hat = var_hat*eta3

    if plot_likelihood:
        plt.figure()
        plt.plot(likelihood)
        plt.xlabel('log likelihood')
        plt.ylabel('iteration')
        plt.show()

    def processInput(i):
        y, z, a, b, muL_hat, muR_hat, var_hat, j, use_tdist = i
        return get_p_val(y, z, a, b, muL_hat, muR_hat, var_hat,
                         ind=j, use_tdist=use_tdist)

    inputs = [(y, z, a, b, muL_hat, muR_hat, var_hat, j, use_tdist)
              for j in genes_to_test_filt]
    p_tn = Parallel(n_jobs=num_cores, verbose=verbose)(delayed(processInput)(i) for i in inputs)
            
    # Add in removed genes due to having 0 var
    j = 0
    p_tn_out = np.ones(len(genes_to_test))
    a_out = np.zeros(len(genes_to_test))
    for i in range(len(genes_to_test)):
        if genes_to_test[i] in d:
            p_tn_out[i] = p_tn[j]
            a_out[i] = a[j]
            j += 1

    if verbose:
        print('')

    output = [p_tn_out]

    if return_split:
        output.append(y_out)
        output.append(z_out)

    if return_likelihood:
        output.append(likelihood)
        
    if return_hyperplane:
        output.append(a_out)
        output.append(b)
        
    if return_consistency:
        output.append(consistency)

    if len(output) == 1:
        return p_tn_out
    
    return output
