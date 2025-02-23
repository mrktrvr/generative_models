import numpy as np
from gamma import Gamma


class Poisson():

    def __init__(self, lamb_a, lamb_b):
        self.lamb_dist = Gamma(lamb_a, lamb_b)

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
