import numpy as np
import scipy.stats as stats
from beta import BetaDistribution


class BernoulliDistribution:
    '''
    bern = BernoulliDistribution(alpha=1, beta=1)
    bern.expectation()
    bern.variance()
    bern.sample(10)
    bern.update_posterior(n_successes, n_failures)
    '''

    def __init__(self, alpha: float = 1, beta: float = 1):
        self.mu_dist = BetaDistribution(alpha, beta)

    def pmf(self, x: int) -> float:
        '''
        Compute the Probability Mass Function (PMF) at x.
        '''
        if x not in [0, 1]:
            raise ValueError('x must be 0 or 1.')
        mu = self.expectation()
        return stats.bernoulli.pmf(x, mu)

    def cdf(self, x: int) -> float:
        '''
        Compute the Cumulative Distribution Function (CDF) at x.
        '''
        if x not in [0, 1]:
            raise ValueError('x must be 0 or 1.')
        mu = self.expectation()
        return stats.bernoulli.cdf(x, mu)

    def expectation(self) -> float:
        '''
        Return the mean of the Bernoulli distribution.
        '''
        mu = self.mu_dist.expectation()
        return mu

    def variance(self) -> float:
        '''
        Return the variance of the Bernoulli distribution.
        '''
        mu = self.expectation()
        return mu * (1 - mu)

    def sample(self, size: int = 1) -> np.ndarray:
        '''
        Generate random samples from the Bernoulli distribution.
        '''
        return np.random.binomial(1, self.expectation(), size)

    def update_posterior(self, n_successes: int, n_failures: int):
        '''
        Update the posterior probability using the Beta distribution.
        '''
        if n_successes < 0 or n_failures < 0:
            raise ValueError(
                'n_successes and n_failures must be non-negative.')
        self.mu_dist.update(n_successes, n_failures)

    def __repr__(self):
        ret = f'BernoulliDistribution(mu={self.expectation()})'
        return ret


def main():
    bern = BernoulliDistribution(alpha=1, beta=1)
    np.random.seed(20)
    print('%2d: %.3f' % (-1, bern.expectation()))
    for i in range(20):
        p = np.random.rand(100)
        n_successes = sum(p >= 0.5)
        n_failures = sum(p < 0.5)
        bern.update_posterior(n_successes, n_failures)
        print('%2d: %.3f' % (i, bern.expectation()), n_successes, n_failures)


if __name__ == '__main__':
    main()
