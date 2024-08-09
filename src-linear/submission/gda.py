import numpy as np

class GDA:
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, theta_0=None, verbose=True):
        """
        Args:
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.verbose = verbose

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y by updating
        self.theta.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        n_examples, dim = x.shape
        self.theta  = np.zeros(dim)

        # solve for parameters
        phi = len(y[y==1.0])/n_examples
        mu0 = np.sum(x[y==0.0], axis=0)/len(y[y==0.0])
        mu1 = np.sum(x[y==1.0], axis=0)/len(y[y==1.0])
        sigma = ((x[y==0.0]-mu0).T @ (x[y==0.0]-mu0) \
                 + (x[y==1.0]-mu1).T @ (x[y==1.0]-mu1))/n_examples
        isigma = np.linalg.inv(sigma)

        # update model
        theta_0 = 0.5*(mu0 @ isigma @ mu0 - mu1 @ isigma @ mu1) - np.log((1-phi)/phi)
        theta = isigma @ (mu1-mu0)
        self.theta = np.append(theta_0, theta)
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        return 1 / (1+np.exp(-self.theta.T @ x.T))
        # *** END CODE HERE