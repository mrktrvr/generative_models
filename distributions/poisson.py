import os
import sys
import numpy as np
from scipy.stats import poisson

CDIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(CDIR, '..'))
from distributions.gamma import Gamma


class Poisson():

    def __init__(self, lamb_a: float = 1, lamb_b: float = 1):
        self.lamb_dist = Gamma(lamb_a, lamb_b)

    def pmf(self, x: int) -> float:
        lamb = self.expectation()
        probability = poisson.pmf(x, lamb)
        return probability

    def cdf(self, x: int) -> float:
        '''
        Compute the Cumulative Distribution Function (CDF) at x.
        '''
        if x not in [0, 1]:
            raise ValueError('x must be 0 or 1.')
        lamb = self.expectation()
        return poisson.cdf(x, lamb)

    def update_posterior(self, data):
        self.lamb_dist.update_posterior(data)

    def expectation(self, sampled_param=False):
        if sampled_param:
            lamb = self.lamb_dist.sample()
        else:
            lamb = self.lamb_dist.expectation()
        return lamb

    def sample(self, size=1, sampled_param=False):
        if sampled_param:
            lamb = self.lamb_dist.sample()
            dst = np.random.poisson(lamb, size=size)
        else:
            lamb = self.lamb_dist.expectation()
            dst = np.random.poisson(lamb, size=size)
        return dst

    def hyper_parameters(self):
        dst = {'lambda': self.lamb_dist.hyper_parameters()}
        return dst


def main():
    lamb = 5
    poi = Poisson(lamb, 1)
    x = 3
    print('mean: %.1f, prob of %d: %.3f' % (poi.expectation(), x, poi.pmf(x)))


if __name__ == '__main__':
    main()
