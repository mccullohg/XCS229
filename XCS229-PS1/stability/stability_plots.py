import util
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

Xa, Ya = util.load_csv('ds1_a.csv', add_intercept=True)
Xb, Yb = util.load_csv('ds1_b.csv', add_intercept=True)

fig = plt.figure(figsize = (4, 3))
ax = fig.add_subplot(projection='3d')
ax.scatter(Xa[:,1], Xa[:,2], Ya)
plt.plot([0,1], [1,0], 0, 'k-')
plt.plot([0,1], [1,0], 1, 'k-')
ax.set_title('Dataset A')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')
plt.show()

fig = plt.figure(figsize = (4, 3))
ax = fig.add_subplot(projection='3d')
ax.scatter(Xb[:,1], Xb[:,2], Yb)
plt.plot([0,1], [1,0], 0, 'k-')
plt.plot([0,1], [1,0], 1, 'k-')
ax.set_title('Dataset B')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')
plt.show()

# Important note: you do not have to modify this file for your homework.

# import util
# import numpy as np


# def calc_grad(X, Y, theta):
#     """Compute the gradient of the loss with respect to theta."""
#     count, _ = X.shape

#     probs = 1. / (1 + np.exp(-X.dot(theta)))
#     grad = (Y - probs).dot(X)

#     return grad


# def logistic_regression(X, Y):
#     """Train a logistic regression model."""
#     theta = np.zeros(X.shape[1])
#     learning_rate = 0.1  # original is 0.1
#     grads = np.zeros((40,1))

#     i = 0
#     while True:
#         i += 1
#         prev_theta = theta
#         grad = calc_grad(X, Y, theta)
#         theta = theta + learning_rate * grad
#         if i % 10000 == 0:
#             print('Finished %d iterations' % i)
#             grads[int(i/10000-1)] = np.linalg.norm(grad, ord=2)
#         if i % 400000 == 0:
#             plt.plot(np.arange(1,41), grads[0:40])
#             plt.title('L2 Norm of the Gradient of the Loss Function')
#             plt.xlabel(r'Iteration ($x10^4$)')
#             plt.ylabel(r'$\left|\nabla L(\theta)\right|_2$')
#             plt.show()
#             break
#         if np.linalg.norm(prev_theta - theta) < 1e-15:
#             print('Converged in %d iterations' % i)
#             break
#     return


# def main():
#     print('==== Training model on data set A ====')
#     Xa, Ya = util.load_csv('ds1_a.csv', add_intercept=True)
#     logistic_regression(Xa, Ya)

#     print('\n==== Training model on data set B ====')
#     Xb, Yb = util.load_csv('ds1_b.csv', add_intercept=True)
#     logistic_regression(Xb, Yb)


# if __name__ == '__main__':
#     main()
