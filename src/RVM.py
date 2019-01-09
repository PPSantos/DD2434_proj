import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel

class RVM:

    def __init__(self):

        # Prior variance.
        self.alphas = None
        # Likelihood variance.
        self.sigma = 0.01**2

        self.phi = None
        self.T = None
        self.X_train = None

        self.threshold_alpha = 1e5
        
        self.removed_bias = False


    def kernel(self, x, y):
        #return linear_kernel(x, y)
        #return rbf_kernel(x, y, 0.1)
        return self.linear_spline_kernel(x, y)
    
    def linear_spline_kernel(self, X, Y):
        phi = np.zeros((X.shape[0], Y.shape[0]))
        
        for i in range(phi.shape[0]):
            for j in range(phi.shape[1]):
                x = X[i]
                y = Y[j]
                phi[i,j] = 1 + x*y + x*y*min(x,y) - ((x+y)*min(x,y)**2)/2 + np.power(min(x,y),3)/3
        
        return phi
    #def linear_kernel(self, x, y):
    #    return np.dot(x, y.T)


    def fit(self, X, T):

        N = X.shape[0]
        self.alphas = 1e-6 * np.ones(N+1)
        self.X_train = X

        # Calculate phi matrix.
        phi = self.kernel(X,X)
        bias_trick = np.ones((N,1))
        self.phi = np.hstack((bias_trick, phi))
        self.T = T

        self.em()

    def calc_posterior(self):

        if self.phi.any == None or self.alphas.any == None or self.sigma == None:
            raise ValueError('Uninitialized attributes.')

        phi = self.phi
        alphas = self.alphas
        sigma = self.sigma

        sigma_posterior = np.linalg.inv((1/sigma) * np.dot(phi.T, phi) + np.diag(alphas))
        mu_posterior = (1/sigma) * np.dot(sigma_posterior, np.dot(phi.T, T))
        return sigma_posterior, mu_posterior

    def prune(self):
        mask = self.alphas < self.threshold_alpha

        # Not remove bias.
        #mask[0] = True

        self.alphas = self.alphas[mask]
        self.phi = self.phi[:, mask]
        
        if not self.removed_bias:
            self.X_train = self.X_train[mask[1:]]
        else:
            self.X_train = self.X_train[mask]
            
        if not mask[0] and not self.removed_bias:
            self.removed_bias = True
            print("bias removed")
        
            
    def em(self):
        
        while(True):
            
            old_alphas = np.copy(self.alphas)
            
            sigma_posterior, mu_posterior = self.calc_posterior()

            gammas = 1 - self.alphas * np.diag(sigma_posterior)
            self.alphas = gammas / (mu_posterior**2)

            # Update sigma.
            '''N = self.alphas.shape[0] - 1
            self.sigma = (N - np.sum(gammas))/(
                    np.sum((T - np.dot(self.phi, mu_posterior)) ** 2))
            #print(self.sigma)
            '''
            
            difference = np.amax(np.abs(self.alphas - old_alphas))
            
            if difference < 1e-3:
                print("EM finished")
                break

            self.prune()
            
    def get_relevance_vectors(self):
        return self.X_train


    def predict(self, X):

        sigma_posterior, mu_posterior = self.calc_posterior()
        
        nb_relev_vect = mu_posterior.shape[0]
        if not self.removed_bias :
            nb_relev_vect -= 1
        print("Nb relevance vectors : ", nb_relev_vect)

        phi = self.kernel(X, self.X_train)
        
        if not self.removed_bias:
            bias_trick = np.ones((N,1))
            phi = np.hstack((bias_trick, phi))

        y = np.dot(phi, mu_posterior)
        
        """if eval_MSE:
            MSE = (1/self.beta_) + np.dot(phi, np.dot(self.sigma_, phi.T))
            return y, MSE[:, 0]
        else:
            return y"""
        
        return y

# test.
N = 500

X = np.linspace(-10,10,N)
#Y = 2*X + 100
Y = np.sinc(X)
T = Y + np.random.randn(N) * 0.01**2

X = X.reshape(N,1)

#plt.scatter(X,T)
#plt.show()

plt.scatter(X, T)
plt.show()

rvm = RVM()

rvm.fit(X,T)

relevance_vect = rvm.get_relevance_vectors()
indexes_relev_vect = [i for i,x in enumerate(X) if x in relevance_vect]

y_pred = rvm.predict(X)



plt.plot(X, y_pred, c='r')
plt.scatter(X, T)
#plt.scatter(relevance_vect, T[indexes_relev_vect], c='g', marker='X', linewidth=12)
plt.show()
