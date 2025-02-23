'''
poisson_mixture.py
'''
import os
import sys
import numpy as np
from scipy import stats

cdir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(cdir, '..'))
from distribution import Gamma
from distribution import Dirichlet
from utils.calc_utils import logsumexp
from utils.logger import logger


def normalize_probs(probs):
    """Normalize probabilities to sum to 1"""
    return probs / np.sum(probs)


class PoissonMixtureModel:

    def __init__(self, pi_alpha, lamb_a, lamb_b):
        '''
        Bayesian Poisson Mixture Model
        Parameters:
        alpha : float
            Dirichlet prior concentration parameter
        gamma_params : tuple
            (a, b) parameters for Gamma prior on lambda
        '''
        self.pi = Dirichlet(pi_alpha)
        self.lamb = Gamma(lamb_a, lamb_b)
        self.lamb_a_prior = lamb_a
        self.lamb_b_prior = lamb_b
        self.n_states = self.pi.n_states

    def e_step(self, X):
        '''Expectation step: compute responsibilities'''
        n_samples = X.shape[0]
        X_flat = X.ravel() if X.ndim > 1 else X
        log_resp = np.zeros((n_samples, self.n_states))
        expt_lamb = self.lamb.expectation()
        for k in range(self.n_states):
            term1 = np.log(self.pi.alpha_post[k])
            term2 = stats.poisson.logpmf(X_flat, expt_lamb[k])
            log_resp[:, k] = term1 + term2
        # Normalize responsibilities
        log_norm = logsumexp(log_resp, axis=1)
        self.resp = np.exp(log_resp - log_norm[:, np.newaxis])
        neg_log_lkh = -np.sum(log_norm)
        return neg_log_lkh

    def m_step(self, X):
        '''
        Maximization step: update parameters
        '''
        sum_s = np.sum(self.resp, axis=0)
        # --- Update weights
        X_flat = X.ravel() if X.ndim > 1 else X
        pi_alpha = self.pi.alpha_prior
        alpha_post = normalize_probs(sum_s + pi_alpha - 1)
        self.pi.alpha_post = alpha_post
        # --- Update lambdas (closed-form solution with Gamma prior)
        sum_s_x = np.sum(self.resp * X_flat[:, np.newaxis], axis=0)
        a_post = sum_s_x + self.lamb.a_prior - 1
        b_post = sum_s + self.lamb.b_prior
        self.lamb.a_post[:] = a_post[:]
        self.lamb.b_post[:] = b_post[:]

    def fit(self, X, init_s=None, max_iter=100, tol=1e-4):
        '''
        Fit the model using EM algorithm
        '''
        X = np.asarray(X)
        if X.ndim == 1:
            X = X[:, np.newaxis]
        if init_s is None:
            n_samples = X.shape[0]
            self.resp = np.zeros((n_samples, self.n_states))
            order = ['s', 'm']
        else:
            self.resp = init_s
            order = ['m', 's']
        log_likelihood_old = None
        for iteration in range(max_iter):
            for em in order:
                if em == 'm':
                    self.m_step(X)
                else:
                    log_likelihood = self.e_step(X)
            if log_likelihood_old is not None:
                if abs(log_likelihood - log_likelihood_old) < tol:
                    break
            log_likelihood_old = log_likelihood

        return self

    def predict(self, X):
        '''Predict cluster assignments'''
        X = np.asarray(X)
        if X.ndim == 1:
            X = X[:, np.newaxis]
        self.e_step(X)
        return np.argmax(self.resp, axis=1)


def generate_synthetic_data(n_samples=300, true_lambdas=[2.0, 10.0, 20.0]):
    '''Generate synthetic Poisson mixture data'''
    weights = [0.3, 0.4, 0.3]
    labels = np.random.choice(len(weights), size=n_samples, p=weights)
    data = np.array([np.random.poisson(true_lambdas[i]) for i in labels])
    return data


def plotter(X, model):
    import matplotlib.pyplot as plt
    x_range = np.arange(0, max(X) + 1)
    mixture_pdf = np.zeros_like(x_range, dtype=float)
    n_states = model.n_states
    cmap = plt.get_cmap('jet')
    cols = [cmap(i / (n_states - 1)) for i in range(n_states)]
    fig = plt.figure(figsize=(12, 4))
    # --- posterior
    pis = [model.pi.expectation(), model.pi.alpha_prior]
    lambs = [model.lamb.expectation(), model.lamb.a_prior / model.lamb.b_prior]
    txt_prm = dict(ha='center', va='top')
    for i, (pi, lamb) in enumerate(zip(pis, lambs), 1):
        ax = fig.add_subplot(1, 2, i)
        ax.hist(X, bins=30, density=True, alpha=0.5, label='Data')
        for k in range(n_states):
            component = (pi[k] * stats.poisson.pmf(x_range, lamb[k]))
            label = ' '.join([
                'state %d' % k,
                r'$\pi$:%.2f' % pi[k],
                r'$\lambda$:%.2f' % lamb[k],
            ])
            ax.plot(x_range, component, label=label, color=cols[k])
            ax.text(lamb[k], 0, '%d' % lamb[k], color=cols[k], **txt_prm)
            mixture_pdf += component
        ax.plot(x_range, mixture_pdf, 'k--', label='Mixture')
        ax.legend()
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.set_title('Bayesian Poisson Mixture Model')
    plt.show()


def main():
    # Generate synthetic data
    np.random.seed(42)
    X = generate_synthetic_data()

    # Fit model
    n_states = 3
    model = PoissonMixtureModel(
        pi_alpha=np.ones(n_states),
        lamb_a=np.ones(n_states),
        lamb_b=np.ones(n_states),
    )
    model.fit(X)

    # Print results
    print('Estimated mixing weights:', model.pi.expectation())
    print('Estimated lambdas:', model.lambdas)

    # Plot results
    plotter(X, model)


if __name__ == '__main__':
    main()
