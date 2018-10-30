from __future__ import print_function

import time
import numpy as np
from scipy.stats import norm, t, truncnorm
from scipy.integrate import quad
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression

# ------------------------------------------------------------------------------
# Basic functions
# ------------------------------------------------------------------------------

def phi(z): return norm.pdf(z)
def Phi(z): return norm.cdf(z)


def trunccdf(x, a, b, mu, var=1):
    """CDF of a truncated normal distribution"""
    xi = (x-mu)/np.sqrt(var)
    alpha = (a-mu)/np.sqrt(var)
    beta = (b-mu)/np.sqrt(var)
    Z = Phi(beta)-Phi(alpha)
    return (Phi(xi)-Phi(alpha))/Z
    

def truncstats(a, b, mu, var=1):
    """Mean and variance of a truncated normal distribution"""
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


def get_null_trunc_params(muL, muR, var=1, a=0, return_LR_params=False):
    """Gets first and second order moments of X and W assuming theta=0"""
    mu = (muL+muR)/2
    muY, varY = truncstats(-np.inf, a, mu, var=var)
    muZ, varZ = truncstats(a, np.inf, mu, var=var)
    
    if return_LR_params:
        return muY, varY, muZ, varZ
    
    muX = muY+muZ
    varX = varY+varZ
    muW = muZ-muY
    varW = varY+varZ
    covXW = varZ-varY
    return muX, varX, muW, varW, covXW 


def get_marginals(a, muL, muR, var=None, ind=0):
    """Gets the marginal distributions of a gene given a as the truncation plane
        
       var can either be None, in which case it's set to all ones, or 
       a vector of estimated variances for each gene (diagonal covariance assumption)
    """
    
    if var is None:
        var = np.ones(len(a))

    # Separating coefficient of interest from other coefficients
    v_in, v_out = var[ind], np.delete(var, ind)
    a_in, a_out = a[ind], np.delete(a, ind)
    muL_in, muL_out = muL[ind], np.delete(muL, ind)
    muR_in, muR_out = muR[ind], np.delete(muR, ind)
    
    def pdfY(Y):
        C = 1/(Phi(np.dot(-a, muL)/np.sqrt(np.sum(a**2*var)))*np.sqrt(2*np.pi*v_in))
        if len(a_out) == 0:
            PhiY = lambda x: 1 if x <= 0 else 0
        else:
            PhiY = lambda x: Phi((-a_in*x-np.dot(a_out, muL_out))/np.sqrt(np.sum(a_out**2*v_out)))
        return C*np.exp(-(Y-muL_in)**2/(2*v_in))*PhiY(Y)

    def pdfZ(Z):
        C = 1/(Phi(np.dot(a, muR)/np.sqrt(np.sum(a**2*var)))*np.sqrt(2*np.pi*v_in))
        if len(a_out) == 0:
            PhiZ = lambda x: 1 if x > 0 else 0
        else:
            PhiZ = lambda x: Phi((a_in*x+np.dot(a_out, muR_out))/np.sqrt(np.sum(a_out**2*v_out)))
        return C*np.exp(-(Z-muR_in)**2/(2*v_in))*PhiZ(Z)
    
    return pdfY, pdfZ


def get_truncmv_params(a, muL, muR, var=None, ind=0):
    """Gets first and second moments"""
    
    pdfY, pdfZ = get_marginals(a, muL, muR, var=var, ind=ind)
    muY = quad(lambda x: x*pdfY(x), -np.inf, np.inf)[0]
    varY = quad(lambda x: x**2*pdfY(x), -np.inf, np.inf)[0]-muY**2
    muZ = quad(lambda x: x*pdfZ(x), -np.inf, np.inf)[0]
    varZ = quad(lambda x: x**2*pdfZ(x), -np.inf, np.inf)[0]-muZ**2
    return muY, varY, muZ, varZ


def get_null_truncmv_params(a, muL, muR, var=None, ind=0, return_LR_params=False):
    """Gets first and second order moments of X and W assuming theta=0
       (multivariate case)
    """
    
    mu = (muL[ind]+muR[ind])/2.
    muL_, muR_ = np.array(muL), np.array(muR)
    muL_[ind], muR_[ind] = mu, mu
    muY, varY, muZ, varZ = get_truncmv_params(a, muL_, muR_, var=var, ind=ind)

    if return_LR_params:
        return muY, varY, muZ, varZ
    
    muX = muY+muZ
    varX = varY+varZ
    muW = muZ-muY
    varW = varY+varZ
    covXW = varZ-varY
    return muX, varX, muW, varW, covXW 


# ------------------------------------------------------------------------------
# Estimation of muL, muR using gradient ascent
# ------------------------------------------------------------------------------

def get_natural_params_diagonal(y, z, a, num_iters=1000, learning_rate=0.1, eps=1e-5,
                                verbose=False, num_stability=1e-10, opt_method='adagrad',
                                var=None):
    """ML estimator for the natural parameters of the joint truncated Gaussian model
       Diagonal covariance matrix assumed.
    
       Notes on optimization:
       - Starting learning rate for each parameter proportional to inverse of variance 
         of y in corresponding entry
       - Adaptive parameter tuning using either Adagrad or RMSprop
       - Only make updates to entries of eta1 that result in positive values
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
    
    def log_loss(eta1, eta2, eta3):
        cY = -np.dot(a, eta2/eta1)/np.sqrt(np.dot(a, a/eta1))
        cZ = np.dot(a, eta3/eta1)/np.sqrt(np.dot(a, a/eta1))
        psi = nY*np.dot(eta2, eta2/eta1)/2 + nZ*np.dot(eta3, eta3/eta1)/2 + \
            nY*np.log(Phi(cY)) + nZ*np.log(Phi(cZ)) + (nY+nZ)*np.sum(np.log(1/eta1))/2
        return -0.5*np.dot(eta1, y2+z2) + np.dot(eta2, y1) + np.dot(eta3, z1) - psi
    
    # Learning rate
    learning_rate_ = learning_rate/np.var(np.vstack((y, z)), 0)
    
    # Gradient cache
    cache = {1: np.zeros(k), 2: np.zeros(k), 3: np.zeros(k)}
    
    if verbose:
        start = time.time()

    loss = [log_loss(eta1, eta2, eta3)]
    
    for t in range(num_iters):
        ieta1 = eta1**-1
        sigma = np.sqrt(np.dot(a, ieta1*a))
        cY = np.dot(-a, ieta1*eta2)/sigma
        cZ = np.dot(a, ieta1*eta3)/sigma
        
        # gradient updates
        deta2 = y1 - nY*ieta1*eta2 + nY*phi(cY)*ieta1*a/(sigma*Phi(cY)+num_stability)
        deta3 = z1 - nZ*ieta1*eta3 - nZ*phi(cZ)*ieta1*a/(sigma*Phi(cZ)+num_stability)
        
        if var is None:
            dphi1 = nY*eta2*eta2 + nZ*eta3*eta3 + (nY+nZ)*eta1 \
                    - nY*(phi(cY)/(Phi(cY)+num_stability))*(2*a*eta2/sigma - np.dot(a, ieta1*eta2)/sigma**3*a**2) \
                    - nZ*(phi(cZ)/(Phi(cZ)+num_stability))*(np.dot(a, ieta1*eta3)/sigma**3*a**2 - 2*a*eta3/sigma)
            deta1 = -y2 - z2 + ieta1*dphi1*ieta1
            deta1 /= 2.
            
        # Adagrad
        if opt_method is 'adagrad':
            if var is None: 
                cache[1] += deta1**2
            cache[2] += deta2**2
            cache[3] += deta3**2
        
        # RMSprop
        elif opt_method is 'RMSprop':
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
        
        loss.append(log_loss(eta1, eta2, eta3))
        
        # check convergence
        if np.abs(loss[-1]-loss[-2]) < eps:            
            break

        if verbose:
            print('\r%s iterations done; loss = %.2e (%.2f s elapsed).'\
                  %(t+1, loss[-1], time.time()-start), end='')

    if verbose:
        print('')
        
    return eta1, eta2, eta3, loss


def get_natural_params_1D(y, z, a, num_iters=1000, learning_rate=0.1, eps=1e-5,
                          num_stability=1e-10, verbose=True, opt_method='adagrad',
                          var=None):
    """ML estimator for the natural parameters of the joint truncated Gaussian model
       1D case
    """
    
    assert isinstance(a, int) or isinstance(a, float)
    assert isinstance(y[0], int) or isinstance(y[0], float)
    assert isinstance(z[0], int) or isinstance(z[0], float)
    
    nY = len(y)
    nZ = len(z)
    learning_rate /= np.max((nY, nZ))
    
    # first and second moments from data
    y1 = np.sum(y)
    z1 = np.sum(z)
    y2 = np.sum(y**2)
    z2 = np.sum(z**2)
    
    def log_loss(eta1, eta2, eta3):
        cY = a*np.sqrt(eta1)-eta2/np.sqrt(eta1)
        cZ = eta3/np.sqrt(eta1)-a*np.sqrt(eta1)
        psi = nY*eta2**2/(2*eta1) + nZ*eta3**2/(2*eta1) + nY*np.log(Phi(cY)) + nZ*np.log(Phi(cZ))-(nY+nZ)*np.log(eta1)/2
        return -0.5*eta1*(y2+z2) + eta2*y1 + eta3*z1 - psi    
    
    start = time.time()
    
    while True:
    
        # initialize parameters
        eta1 = 1./var if var is not None else np.random.uniform(0.5, 1)
        eta2 = np.random.randn()
        eta3 = np.random.randn()

        loss = [log_loss(eta1, eta2, eta3)]

        # Gradient cache
        cache = np.zeros(3)

        for t in range(num_iters):
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
            if opt_method is 'adagrad':
                if var is None:
                    cache[0] += deta1**2
                cache[1] += deta2**2
                cache[2] += deta3**2

            # RMSprop
            elif opt_method is 'RMSprop':
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
                if eta1_ > 0: eta1 = eta1_ # ensure that eta1 stays positive
            eta2 += learning_rate*deta2/(np.sqrt(cache[1])+num_stability)
            eta3 += learning_rate*deta3/(np.sqrt(cache[2])+num_stability)

            loss.append(log_loss(eta1, eta2, eta3))
            
            # check convergence
            if np.abs(loss[-1]-loss[-2]) < eps or np.isnan(loss[-1]):            
                break
                
            if verbose:
                print('\r%s/%s gradient udpates performed (%.2f s elapsed).'
                      %(t+1, num_iters, time.time()-start), end='')
            
        if not np.isnan(loss[-1]):
            break
            
    if verbose:
        print('')
        
    return eta1, eta2, eta3, loss


# ------------------------------------------------------------------------------
# Compute corrected p value
# ------------------------------------------------------------------------------

def get_p_val(y, z, a, muL, muR, var, ind=0, use_tdist=False):
    """
    Correct pval approach using approximations of truncated Gaussians
    
    Parameters
    ----------
    y: points from one cluster
    z: points from the other cluster
    a: threshold
    use_tdist: False for standard normal
               True for t distribution with df=len(y)+len(z)-2
    muL, muR, var: estimated using maximum likelihood
    ind: gene to test
    
    Returns
    ----------
    pvalue
    """
    muY, varY, muZ, varZ = get_null_truncmv_params(a, muL, muR, var=var,
                                                   ind=ind, return_LR_params=True)
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
    (y, z, a are each scalars)
    
    Parameters
    ----------
    y: points from one cluster
    z: points from the other cluster
    a: threshold
    use_tdist: False for standard normal
               True for t distribution with df=len(y)+len(z)-2
    muL, muR, var: estimated using maximum likelihood
    
    Returns
    ----------
    pvalue
    """
    muY, varY, muZ, varZ = get_null_trunc_params(muL, muR, var=var, a=a,
                                                 return_LR_params=True)
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
# TN tests
# ------------------------------------------------------------------------------

def split_and_est_a(y, z, verbose=False, split_prop=0.5, zero_mean=True,
                    method='linear', C=100):
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
    if zero_mean:
        data1 -= np.mean(data1, 0)
        data2 -= np.mean(data2, 0)
        
    # Use first half to fit hyperplane
    if method == 'linear':
        lr = LinearRegression(fit_intercept=False)
        lr.fit(data1, labels1)
        a = lr.coef_.reshape(-1)
        labels2 = (np.dot(data2, a) > 0).astype(int)
        
    elif method == 'svm':
        svm = SVC(kernel='linear', C=C)
        svm.fit(data1, labels1)
        a = svm.coef_.reshape(-1)
        labels2 = svm.predict(data2)
        
    y = data2[labels2 == 0]
    z = data2[labels2 == 1]
    if verbose: 
        print('Labels assigned.. (%.2f s elapsed)'%(time.time()-start))
        print('Consistency of new labels with old: %.3f'\
              %(np.sum(labels2 == labels2true)/float(len(labels2))))
        
    return y, z, a


def tn_test(y, z, a=None, genes_to_test=None, var=None, verbose=False, 
            return_split=False, split_prop=0.5, zero_mean=True, method='svm',
            num_iters=100000, learning_rate=0.1, eps=1e-5, use_tdist=False):
    
    if verbose:
        start = time.time()
    
    if a is None:
        y, z, a = split_and_est_a(y, z, verbose=verbose, split_prop=split_prop,
                                  zero_mean=zero_mean, method=method)
    
    if genes_to_test is None:
        genes_to_test = range(len(a))
        
    if verbose:
        print('Number of genes with 0 variance across cells: %s'\
              %(np.sum(np.var(np.vstack((y, z)), 0) == 0)))

    eta1, eta2, eta3, track = get_natural_params_diagonal(y, z, a, var=var,
                                                          num_iters=num_iters,
                                                          learning_rate=learning_rate, 
                                                          verbose=verbose,
                                                          eps=eps)
    var_hat = eta1**-1
    muL_hat = var_hat*eta2
    muR_hat = var_hat*eta3
    
    
    p_tn = np.zeros(len(genes_to_test))
    for i, j in enumerate(genes_to_test):
        p_tn[i] = get_p_val(y, z, a, muL_hat, muR_hat, var_hat,
                            ind=j, use_tdist=use_tdist)
        if verbose:
            print('\r%s/%s genes tested (%.2f s elapsed)'\
                  %(i+1, len(genes_to_test), time.time()-start), end='')

    if verbose:
        print('')
        
    if return_split:
        return p_tn, y, z
    
    return p_tn


def tn_cluster_test(y, z, plot_proj=False, learning_rate=10., use_tdist=False,
                    num_iters=100000, verbose=False, zero_mean=True, method='linear'):
    """Test significance of clusters
       (y, z are two different groups)
    """
    
    data = np.vstack((y, z))
    labels = np.hstack((np.zeros(len(y)), np.ones(len(z))))
    
    # Split dataset in half
    inds1 = np.zeros(len(data)).astype(bool)
    samp = np.sort(np.random.choice(range(len(data)), int(len(data)/2), replace=False))
    inds1[samp] = True
    data1 = data[inds1]
    labels1 = labels[inds1]
    data2 = data[~inds1]
    
    if zero_mean:
        data1 -= np.mean(data1, 0)
        data2 -= np.mean(data2, 0)

    # Use first half to fit hyperplane
    if method == 'linear':
        lr = LinearRegression(fit_intercept=False)
        lr.fit(data1, labels1)
        a = lr.coef_.reshape(-1)

        # Assign labels for second half, the part used for testing
        labels2 = (np.dot(data2, a) > 0).astype(int)
        
    elif method == 'svm':
        svm = SVC(kernel='linear', C=100.)
        svm.fit(data1, labels1)
        a = svm.coef_.reshape(-1)
        labels2 = svm.predict(data2)
     
    y2 = data2[labels2 == 0]
    z2 = data2[labels2 == 1]

    # Project X2 along separating hyperplane
    u = a/np.linalg.norm(a)
    y2_proj = np.dot(y2, u)
    z2_proj = np.dot(z2, u)
    
    if plot_proj:
        plt.figure(figsize=(5, 2.5))
        tn.plot_stacked_hist(y2_proj, z2_proj)
        plt.show()

    # Parameter estimation
    eta1, eta2, eta3, track = get_natural_params_diagonal(y2_proj.reshape(-1, 1),
                                                          z2_proj.reshape(-1, 1), 
                                                          np.ones(1), 
                                                          num_iters=num_iters,
                                                          eps = 1e-3,
                                                          learning_rate=learning_rate, 
                                                          verbose=verbose)
    var_hat = eta1**-1
    muL_hat = var_hat*eta2
    muR_hat = var_hat*eta3
    
    if verbose:
        print('\rvar = %s, muL = %s, muR = %s'%(var_hat, muL_hat, muR_hat))
    
    p = get_p_val(y2_proj.reshape(-1, 1), z2_proj.reshape(-1, 1), np.ones(1),
                  muL_hat, muR_hat, var_hat, use_tdist=use_tdist)

    return p