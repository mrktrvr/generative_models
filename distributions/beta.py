import numpy as np
import scipy.stats as stats


class BetaDistribution:

    def __init__(self, alpha: float, beta: float):
        if alpha <= 0 or beta <= 0:
            raise ValueError('Alpha and Beta must be positive.')
        self.alpha = alpha
        self.beta = beta
        self.prior_alpha = alpha
        self.prior_beta = beta

    def pdf(self, x: float) -> float:
        '''
        Compute the Probability Density Function (PDF) at x.
        '''
        if x < 0 or x > 1:
            raise ValueError('x must be in the range [0, 1].')
        return stats.beta.pdf(x, self.alpha, self.beta)

    def cdf(self, x: float) -> float:
        '''
        Compute the Cumulative Distribution Function (CDF) at x.
        '''
        if x < 0 or x > 1:
            raise ValueError('x must be in the range [0, 1].')
        return stats.beta.cdf(x, self.alpha, self.beta)

    def mean(self) -> float:
        '''
        Return the mean of the Beta distribution.
        '''
        return self.alpha / (self.alpha + self.beta)

    def variance(self) -> float:
        '''
        Return the variance of the Beta distribution.
        '''
        total = self.alpha + self.beta
        return (self.alpha * self.beta) / (total**2 * (total + 1))

    def sample(self, size: int = 1) -> np.ndarray:
        '''
        Generate random samples from the Beta distribution.
        '''
        return np.random.beta(self.alpha, self.beta, size)

    def update(self, successes: int, failures: int):
        '''
        Update the Beta distribution parameters based on observed data.

        success: number of positive samples (number of head in coin trial)
        failures: number of negative samples (number of tail in coin trial)
        success + failure = number of trial
        '''
        if successes < 0 or failures < 0:
            raise ValueError('Successes and failures must be non-negative.')
        self.alpha += successes
        self.beta += failures

    def __repr__(self):
        ret = f'BetaDistribution(alpha={self.alpha}, beta={self.beta})'
        return ret


def main():
    beta = BetaDistribution(10, 10)
    print('%2d: %.3f' % (-1, beta.mean()))
    np.random.seed(20)
    for i in range(20):
        p = np.random.rand(100)
        success = sum(p >= 0.5)
        failure = sum(p < 0.5)
        beta.update(success, failure)
        print('%2d: %.3f' % (i, beta.mean()), success, failure)


if __name__ == '__main__':
    main()
