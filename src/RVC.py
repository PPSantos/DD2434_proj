import numpy as np 
import matplotlib.pyplot as plt
from scipy.special import expit
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel, polynomial_kernel

from tqdm import tqdm_notebook as tqdm

from scipy.optimize import minimize

class RVC:
    """

        Relevance Vector Machine - Classification
        
    """
    def __init__(
        self,
        kernel='rbf',
        coef0=None,
        em_tol=1e-3,
        alpha=1e-6,
        threshold_alpha=1e9,
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
        
    def classify(self, mu_posterior, phi):
        return expit(np.dot(phi, mu_posterior))

    '''def calc_posterior(self):
        """
            Calculate posterior.
        """
        
        phi = self.phi
        alphas = self.alphas
        T = self.T
        mu_posterior = self.mu_posterior
    
        
        y = self.classify(mu_posterior, phi)
        B = np.diag(y*(1-y)) 

        sigma_posterior = np.linalg.inv(np.diag(alphas) + np.dot(phi.T, np.dot(B, phi)))
        self.mu_posterior = np.dot(np.dot(sigma_posterior, np.dot(phi.T, B)), T)

        return sigma_posterior'''
    
    def log_posterior(self, mu_posterior, alphas, phi, T):

        y = self.classify(mu_posterior, phi)

        log_p = -1 * (np.sum(np.log(y[T == 1]), 0) +
                      np.sum(np.log(1-y[T == 0]), 0))
        log_p = log_p + 0.5*np.dot(mu_posterior.T, np.dot(np.diag(alphas), mu_posterior))

        jacobian = np.dot(np.diag(alphas), mu_posterior) - np.dot(phi.T, (T-y))

        return log_p, jacobian

    def hessian(self, mu_posterior, alphas, phi, T):
        y = self.classify(mu_posterior, phi)
        B = np.diag(y*(1-y))
        return np.diag(alphas) + np.dot(phi.T, np.dot(B, phi))

    def posterior(self):
        result = minimize(
            fun=self.log_posterior,
            hess=self.hessian,
            x0=self.mu_posterior,
            args=(self.alphas, self.phi, self.T),
            method='Newton-CG',
            jac=True,
            options={
                'maxiter': 50
            }
        )

        self.mu_posterior = result.x
        sigma_posterior = np.linalg.inv(
            self.hessian(self.mu_posterior, self.alphas, self.phi, self.T)
        )
        
        return sigma_posterior

    def prune(self):
        """
            Pruning based on alpha values.
        """
        mask = self.alphas < self.threshold_alpha

        self.alphas = self.alphas[mask]
        self.old_alphas = self.old_alphas[mask]
        self.phi = self.phi[:, mask]
        self.mu_posterior = self.mu_posterior[mask]
        
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
        #while(True):
        for _ in tqdm(range(10000)):
            
            self.old_alphas = np.copy(self.alphas)
            
            #sigma_posterior = self.calc_posterior()
            sigma_posterior = self.posterior()
            

            gammas = 1 - self.alphas * np.diag(sigma_posterior)
            self.alphas = gammas / (self.mu_posterior**2)
            
            self.prune()
            #print("relev : ", self.relevance_vec.shape)

            difference = np.amax(np.abs(self.alphas - self.old_alphas))
            
            #print("diff : ", difference)
            if difference < self.em_tol:
                if self.verbose:
                    print("EM finished")
                break


    def predict_proba(self, X):
        
        phi = self.kernel(X, self.relevance_vec)
        
        if not self.removed_bias:
            bias_trick = np.ones((X.shape[0], 1))
            phi = np.hstack((bias_trick, phi))
            
        y = self.classify(self.mu_posterior, phi)
        
        return y
    
    def predict(self, X):
        
        y = self.predict_proba(X)
        pred = np.where(y > 0.5, 1, 0)
        
        return pred
