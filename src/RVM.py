import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import linear_kernel

class RVM:

    def __init__(self):

        # Prior variance.
        self.alphas = None
        # Likelihood variance.
        self.sigma = 1

        self.phi = None
        self.T = None


    def kernel(self, x, y):
        return linear_kernel(x, y)

    #def linear_kernel(self, x, y):
    #    return np.dot(x, y.T)


    def fit(self, X, T):

        N = X.shape[0]
        self.alphas = 1 * np.ones(N+1)

        # Calculate phi matrix.
        phi = self.kernel(X,X)
        bias_trick = np.ones((N,1))
        self.phi = np.hstack((bias_trick, phi))
        self.T = T

        self.em()

    def calc_posterior(self):

        if self.phi.any == None or self.alphas.any == None or self.sigma == None:
            raise ValueError('Not initialized attributes.')

        phi = self.phi
        alphas = self.alphas
        sigma = self.sigma

        sigma_posterior = np.linalg.inv((1/sigma) * np.dot(phi.T, phi) + np.diag(alphas))
        mu_posterior = (1/sigma) * np.dot(sigma_posterior, np.dot(phi.T, T))
        return sigma_posterior, mu_posterior


    def em(self):

        #while(True):
        for _ in range(1):

            sigma_posterior, mu_posterior = self.calc_posterior()

            gammas = 1 - np.diag(self.alphas) * sigma_posterior
            self.alphas = gammas / (mu_posterior**2)

            # Update sigma.
            N = self.alphas.shape[0] - 1
            self.sigma = (N - np.sum(gammas))/(
                    np.sum((T - np.dot(self.phi, mu_posterior)) ** 2))

# test.
N = 50

X = np.linspace(0,10,N)
Y = 2*X
T = Y + np.random.randn(N)

Y = Y.reshape(N,1)
X = X.reshape(N,1)

#plt.scatter(X,T)
#plt.show()

rvm = RVM()

rvm.fit(X,T)
