import numpy as np 
import matplotlib.pyplot as plt
from scipy.special import expit
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel, polynomial_kernel

class RVC:

    def __init__(
        self,
        kernel='rbf',
        coef0=0.01,
        em_tol=1e-3,
        alpha=1e-6,
        threshold_alpha=1e5,
        update_sigma=False,
        verbose=False
        ):

        # Kernel function.
        self.kernel_type = kernel

        # Kernel coefficient.
        self.coef0 = coef0

        # EM stop condition tolerance.
        self.em_tol = em_tol

        # Alphas prior variance.
        self.alpha = alpha

        # Alphas pruning threshold.
        self.threshold_alpha = threshold_alpha

        self.update_sigma = update_sigma
        self.verbose = verbose
        
        # Diagonal matrix of sigmoid gradient
        self.B = None

        # Prior variances (weights).
        self.alphas = None
        self.old_alphas = None

        # 'Design matrix'.
        self.phi = None

        # Targets.
        self.T = None

        # Relevance vectors.
        self.relevance_vec = None

        # Number of training points.
        self.N = None
        
        # True if bias was pruned.
        self.removed_bias = False


    def get_relevance_vectors(self):
        return self.relevance_vec


    def kernel(self, x, y):
        """
            Applies kernel function.
        """
        if self.kernel_type == 'rbf':
            return rbf_kernel(x, y, self.coef0)
        elif self.kernel_type == 'linear':
            return linear_kernel(x, y)
        elif self.kernel_type == 'linear_spline':
            return self.linear_spline_kernel(x, y)
        elif self.kernel_type == 'exponential':
            return self.exponential_kernel(x, y)
        else:
            raise ValueError('Undefined kernel.')

    
    def linear_spline_kernel(self, X, Y):
        """
            Linear spline kernel.
        """
        phi = np.zeros((X.shape[0], Y.shape[0]))
        
        for i in range(phi.shape[0]):
            for j in range(phi.shape[1]):
                x = X[i]
                y = Y[j]
                phi[i,j] = 1 + x*y + x*y*min(x,y) - \
                ((x+y)*min(x,y)**2)/2 + np.power(min(x,y),3)/3
        
        return phi
    
    def exponential_kernel(self, X, Y):
        
        phi = np.zeros((X.shape[0], Y.shape[0]))
        
        for i in range(phi.shape[0]):
            for j in range(phi.shape[1]):
                x_1 = X[i, 0]
                x_2 = X[i, 1]
                y_1 = Y[j, 0]
                y_2 = Y[j, 1]
                phi[i,j] = np.exp(-997e-4*(x_1-y_1)**2 - 2e-4*(x_2-y_2)**2)
        
        return phi
        


    def fit(self, X, T):
        """
            Train.
        """

        self.N = X.shape[0]
        self.relevance_vec = X
        self.T = T

        self.alphas = self.alpha * np.ones(self.N+1)

        # Calculate phi matrix and append bias trick column.
        phi = self.kernel(X,X)
        bias_trick = np.ones((self.N,1))
        self.phi = np.hstack((bias_trick, phi))
        
        self.mu_posterior = np.zeros(self.N+1)
        
        self.em()
        
    def classify(self, m, phi):
        return expit(np.dot(phi, m))

    def calc_posterior(self):
        """
            Calculate posterior.
        """

        phi = self.phi
        alphas = self.alphas
        T = self.T
        
        #B = 

        sigma_posterior = np.linalg.inv(np.dot(np.dot(phi.T, B), phi) + np.diag(alphas))
        self.mu_posterior = np.dot(np.dot(sigma_posterior, np.dot(phi.T, B)), T)

        return sigma_posterior, mu_posterior

    def prune(self):
        """
            Pruning based on alpha values.
        """
        mask = self.alphas < self.threshold_alpha

        self.alphas = self.alphas[mask]
        self.old_alphas = self.old_alphas[mask]
        self.phi = self.phi[:, mask]
        
        if not self.removed_bias:
            self.relevance_vec = self.relevance_vec[mask[1:]]
        else:
            self.relevance_vec = self.relevance_vec[mask]
            
        if not mask[0] and not self.removed_bias:
            self.removed_bias = True
            if self.verbose:
                print("Bias removed")
        
            
    def em(self):
        """
            EM.
        """
        while(True):
            
            self.old_alphas = np.copy(self.alphas)
            
            sigma_posterior, mu_posterior = self.calc_posterior()

            gammas = 1 - self.alphas * np.diag(sigma_posterior)
            self.alphas = gammas / (mu_posterior**2)

            # Update sigma.
            if self.update_sigma:
                #N = self.alphas.shape[0] - 1
                self.sigma = (self.N - np.sum(gammas))/(
                        np.sum((self.T - np.dot(self.phi, mu_posterior)) ** 2))
            
            self.prune()

            difference = np.amax(np.abs(self.alphas - self.old_alphas))
            
            if difference < self.em_tol:
                if self.verbose:
                    print("EM finished")
                break


    def predict_proba(self, X):
        
        phi = self.kernel(X, self.relevance_vec)
        y = self.classify(self.mu_posterior, phi)
        
        print(y)
        
        return None
