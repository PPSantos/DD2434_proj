import numpy as np 
import matplotlib.pyplot as plt
#from sklearn.metrics.pairwise import linear_kernel

N = 50

X = np.linspace(0,10,N)
Y = 2*X
T = Y + np.random.randn(N)

Y = Y.reshape(N,1)
X = X.reshape(N,1)

#plt.scatter(X,T)
#plt.show()

# Prior variance.
alphas = 1e-6 * np.ones(N+1)
# Likelihood variance.
sigma = 1

def kernel(x, y):
    return linear_kernel(x, y)

def linear_kernel(x, y):
    return np.dot(x, y.T)

phi = kernel(X,X)
bias_trick = np.ones((N,1))
phi = np.hstack((bias_trick, phi))

sigma_posterior = np.linalg.inv((1/sigma) * np.dot(phi.T, phi) + np.diag(alphas))
mu_posterior = (1/sigma) * np.dot(sigma_posterior, np.dot(phi.T, T))


"""
i_s = np.diag(self.alpha_) + self.beta_ * np.dot(self.phi.T, self.phi)
self.sigma_ = np.linalg.inv(i_s)
self.m_ = self.beta_ * np.dot(self.sigma_, np.dot(self.phi.T, self.y))


"""
