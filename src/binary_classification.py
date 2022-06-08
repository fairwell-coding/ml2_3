""" Python code submission file.

IMPORTANT:
- Do not include any additional python packages.
- Do not change the interface and return values of the task functions.
- Only insert your code between the Start/Stop of your code tags.
- Prior to your submission, check that the pdf showing your plots is generated.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import approx_fprime
from matplotlib.backends.backend_pdf import PdfPages


global data_a_train, data_a_test, data_b_train, data_b_test


def load_data():
    """ General utility function to load provided .npy data

        For datasets A and B:
        - load train/test data (N/N_t samples)
        - the first 2 columns are input x and the last column is target y for all N/N_t samples
        - include transform to homogeneous input data
    """

    """ Start of your code 
    """

    data_a_train, data_a_test = np.load('data_a_test.npy'), np.load('data_a_test.npy')
    data_b_train, data_b_test = np.load('data_b_train.npy'), np.load('data_b_test.npy')

    """ End of your code
    """

    return data_a_train, data_a_test, data_b_train, data_b_test

    
def quadratic():
    """ Subtask 1: Quadratic Loss as Convex Surrogate in Binary Classification

        Requirements for the plot:
        - plot each of the groundtruth test data A and predicted test data A in a scatterplot into one subplot
        - indicate the two classes with 2 different colors for both subplots
        - use a legend
    """


    fig, ax = plt.subplots(1, 2, figsize=(10,5))
    plt.suptitle('Task 1 - Quadratic Loss as Convex Surrogate in Binary Classification', fontsize=12)
    ax[0].set_title('Test Data A')
    ax[1].set_title('Test Predictions A')

    """ Start of your code 
    """
    X_train = np.stack([data_a_train[:, 0], data_a_train[:, 1]]).T # X = 100x2
    Y_train = data_a_train[:, 2] # Y = 100x1

    N_train = len(Y_train)
    M = 1 + X_train.shape[1]

    # calculate weight vector w
    PHI = feature_transform(X_train)
    w = invert(PHI.T@PHI) @ PHI.T @ Y_train
    Y_train_pred = np.sign(PHI @ w.T)
    # w1 = __calculate_w(PHI, Y)
    # w - w1 = [0.0, 0.0, 7.632783294297951e-17]

    X_test = np.stack([data_a_test[:, 0], data_a_test[:, 1]]).T # X = 100x2
    Y_test = data_a_test[:, 2] # Y = 100x1
    N_test = len(Y_test)
    PHI =  feature_transform(X_test)
    Y_test_pred = np.sign(PHI @ w.T)
 
    train_acc, test_acc = len(np.nonzero(Y_train == Y_train_pred)[0]) / N_train, len(np.nonzero(Y_test == Y_test_pred)[0]) / N_test
    print(f'The quadratic problem has training accuracy = {train_acc} and test accuracy = {test_acc}.')

    C1_ind = np.nonzero(Y_test >= 0)
    C2_ind = np.nonzero(Y_test < 0)
    marker_size = 60
    ax[0].scatter(X_test[C1_ind, 0], X_test[C1_ind, 1], marker_size, c='blue', marker='o', label='test data c1 (+)')
    ax[0].scatter(X_test[C2_ind, 0], X_test[C2_ind, 1], marker_size, c='red',  marker='o', label='test data c2 (x)')

    test_indices_C1 = np.nonzero(Y_test_pred >= 0)
    test_indices_C2 = np.nonzero(Y_test_pred < 0)
    ax[1].scatter(X_test[C1_ind, 0], X_test[C1_ind, 1], marker_size, c='blue', marker='o', label='test data c1 (+)')
    ax[1].scatter(X_test[C2_ind, 0], X_test[C2_ind, 1], marker_size, c='red',  marker='o', label='test data c2 (x)')
    
    ax[1].scatter(X_test[test_indices_C1, 0], X_test[test_indices_C1, 1], marker_size, c='k', marker='+', label='test prediction c1')
    ax[1].scatter(X_test[test_indices_C2, 0], X_test[test_indices_C2, 1], marker_size, c='k',  marker='x', label='test prediction c2')

    """ End of your code
    """

    ax[0].legend()
    ax[1].legend()
    plt.show()
    return fig

def feature_transform(X):
    N = X.shape[0]
    return np.stack([np.ones((N,)), X[:, 0], X[:, 1]]).T # PHI = 100x3

# def __calculate_w(theta, Y):
#     A = theta.T @ theta
#     Q, R = np.linalg.qr(A)
#     z = Q.T @ theta.T @ Y
#     w = np.linalg.solve(R, z)
#     return w

def invert(A):
    Q, R = np.linalg.qr(A)
    return np.linalg.inv(R) @ Q.T

def logistic():
    """ Subtask 2: Logistic Loss as Convex Surrogate in Binary Classification

        Requirements for the plot:
        - the first subplot should contain the energy of the objective function plotted over the iterations
        - the other two subplots should contain groundtruth test data A and predictions of test data A, respectively, in a scatterplot
            - indicate the two classes with 2 different colors
            - use a legend
    """
    fig, ax = plt.subplots(1, 3, figsize=(15,5))
    plt.suptitle('Task 2 - Logistic Loss as Convex Surrogate in Binary Classification', fontsize=12)

    ax[0].set_title('Energy $E(\widetilde{\mathbf{w}})$')
    ax[1].set_title('Test Data A')
    ax[2].set_title('Test Predictions A')

    """ Start of your code 
    """
    X_train = np.stack([data_a_train[:, 0], data_a_train[:, 1]]).T # X = 100x2
    Y_train = data_a_train[:, 2] # Y = 100x1

    N = len(Y_train)
    M = 1 +  X_train.shape[1]

    PHI = feature_transform(X_train)
    L = calculate_lipschitz(PHI)
    w0 = np.ones((M, 1))

    k_max = 1000
    #TODO: calculate engery
    E = np.zeros((k_max, 1))
    w, E = nesterov_gradient(Y_train, X_train, w0, L, k_max)
    # Y_train_pred = np.sign(sigmoid(PHI @ w) - 0.5)
    Y_train_pred = np.where(sigmoid(PHI @ w) < 0.5, -1 ,1)

    X_test = np.stack([data_a_test[:, 0], data_a_test[:, 1]]).T # X = 100x2
    Y_test = data_a_test[:, 2] # Y = 100x1

    N_train = len(Y_train)
    N_test = len(Y_test)

    PHI = feature_transform(X_test)
    Y_test_pred = np.where(sigmoid(PHI @ w) < 0.5, -1 ,1)

    train_acc, test_acc = len(np.nonzero(Y_train.reshape((N_train, 1)) == Y_train_pred)[0]) / N_train, len(np.nonzero(Y_test.reshape((N_test, 1)) == Y_test_pred)[0]) / N_test
    print(f'The logistic regression has training accuracy = {train_acc} and test accuracy = {test_acc}.')

    iters = np.arange(1, len(E))
    ax[0].loglog(iters, E[:-1]-E.min(), color="blue", linewidth=2.0, label="Logistic loss")
    # ax[0].loglog(iters, E.max()/iters**2, color="black", linewidth=2.0, label="$O(1/k^2)$")

    marker_size = 60

    C1_ind = np.nonzero(Y_test == 1)
    C2_ind = np.nonzero(Y_test == -1)
    ax[1].scatter(X_test[C1_ind, 0], X_test[C1_ind, 1], marker_size, c='blue', marker='o', label='test data c1 (+)')
    ax[1].scatter(X_test[C2_ind, 0], X_test[C2_ind, 1], marker_size, c='red',  marker='o', label='test data c2 (x)')

    test_indices_C1 = np.nonzero(Y_test_pred == 1)
    test_indices_C2 = np.nonzero(Y_test_pred == -1)
    ax[2].scatter(X_test[C1_ind, 0], X_test[C1_ind, 1], marker_size, c='blue', marker='o', label='test data c1 (+)')
    ax[2].scatter(X_test[C2_ind, 0], X_test[C2_ind, 1], marker_size, c='red',  marker='o', label='test data c2 (x)')

    ax[2].scatter(X_test[test_indices_C1, 0], X_test[test_indices_C1, 1], marker_size, c='k', marker='+', label='test prediction c1')
    ax[2].scatter(X_test[test_indices_C2, 0], X_test[test_indices_C2, 1], marker_size, c='k',  marker='x', label='test prediction c2')

    """ End of your code
    """

    ax[1].legend()
    ax[2].legend()
    plt.show()
    return fig

def calculate_lipschitz(PHI):
    return 1/4 * np.amax(sigmoid(PHI)) #largest singular value

def loss2(y, y_hat):
    y = np.clip(y+1, 0, 1)
    loss = -np.mean(y*(np.log(y_hat)) - (1-y)*np.log(1-y_hat))
    return loss

def log_loss(X, y, w):
    N = len(y)
    PHI = feature_transform(X)
    return np.log(1 + np.exp(- y * PHI @ w)).sum()

def grad_log_loss(X, y, w):
    N = len(y)
    PHI = X# feature_transform(X.reshape((N, 2)))
    w = w.reshape((3,1))
    y = y.reshape((N, 1))
    return np.sum(PHI.T @ (y * (sigmoid(y * PHI @ w) - 1)), axis=1)

def nesterov_gradient(y, X, w0, L, k_max):

    PHI = feature_transform(X)
    y = y.reshape((100, 1))
    E_ = []
    E_app = np.zeros((3,1))
    w_old = w0
    w = w0
    for k in range(1, k_max+1):
        beta = (k-1)/(k+1)
        w_ = w  + beta * (w - w_old)
        w_old = w
        w = w_ - 1/L * grad_log_loss(PHI, y, w_).reshape((3, 1))
        E_.append(log_loss(X, y, w))
        # E_.append(log_loss(PHI, y, w))
        if (k-1)%10 == 0:      
            E_app = approx_fprime(w.flatten(), lambda w__: log_loss(PHI, y, w__), 1e-8)
            print(f'{log_loss(PHI, y, w)} | {E_app.flatten()} | {grad_log_loss(PHI, y, w).flatten()}')

    return w, np.array(E_)

def svm_primal():
    """ Subtask 3: Hinge Loss as Convex Surrogate in Binary Classification

        Requirements for the plot:
        - the first subplot should contain the energy of the objective function plotted over the iterations
        - the next two subplots should contain predictions of train data A and test data A, respectively, in a scatterplot
            - indicate the two predicted classes with 2 different colors for both data sets 
            - use a legend
            - for both train and test data also plot the separating hyperplane and the margin at \pm 1
            - for the train data include the support vectors
    """
    fig, ax = plt.subplots(1,3, figsize=(15,5))
    plt.suptitle('Task 3 - Hinge Loss as Convex Surrogate in Binary Classification', fontsize=12)

    title_list = ['Energy', 'Train Predictions A', 'Test Predictions A']
    for idx, a in enumerate(ax.reshape(-1)):
        a.set_title(title_list[idx])

    """ Start of your code 
    """

    marker_size = 30
    # load data
    X_train = np.stack([data_a_train[:, 0], data_a_train[:, 1]]).T  # X = 100x2
    y_train = data_a_train[:, 2]  # y = 100x1
    X_test = np.stack([data_a_test[:, 0], data_a_test[:, 1]]).T  # X = 100x2
    y_test = data_a_test[:, 2]  # y = 100x1

    # hyperparameters
    lambda_ = 1e-4  # 1.24
    # alpha = 0.9
    alpha = 0.9
    w = np.ones((3, 1))
    w[0] = 0.0
    delta = 1e-4

    PHI = feature_transform(X_train)
    w, b, loss = __proximal_subgradient_method(PHI, y_train, alpha, lambda_, w, delta)
    # b = np.mean(-(PHI[:, 1:] @ w/ np.linalg.norm(w)))
    w = np.concatenate([b.reshape((1,1)), w])
    #w = w / np.linalg.norm(w)

    support_vectors = y_train.reshape((100,1)) * (feature_transform(X_train) @ w) #+
    y_train_pred = np.sign(feature_transform(X_train) @ w )
    y_test_pred = np.sign(feature_transform(X_test) @ w )


    sv_indices_c1 = np.nonzero(np.isclose(np.round(support_vectors, 2),  1/np.linalg.norm(w), atol=0.1))[0]
    sv_indices_c2 = np.nonzero(np.isclose(np.round(support_vectors, 2), -1/np.linalg.norm(w), atol=0.1))[0]

    train_acc, test_acc = len(np.nonzero(y_train.reshape(y_train.shape[0], 1) == y_train_pred)[0]) / X_train.shape[0], len(np.nonzero(y_test.reshape(y_test.shape[0], 1) == y_test_pred)[0]) / \
                          X_test.shape[0]
    print(f'SVM primal space: train_acc = {train_acc}, test_acc = {test_acc}')

    ax[0].plot(np.arange(0, len(loss))+1, loss)
    ax[0].set_xlabel('iterations')
    ax[0].set_ylabel('Energy')
    __plots_svm_primal(ax[1], PHI, X_train, w, marker_size, sv_indices_c1, sv_indices_c2)
    plot_decision_boundary(X_train, w, ax[1])
    __plots_svm_primal(ax[2], PHI, X_test, w, marker_size, sv_indices_c1, sv_indices_c2)
    plot_decision_boundary(X_train, w, ax[2])
    plt.show()


    """ End of your code
    """

    ax[1].legend()
    ax[2].legend()
    return fig

def __plots_svm_primal(ax, PHI, X_train, w, marker_size, sv_indices_c1, sv_indices_c2):
    
    # plot data points
    ax.scatter(X_train[:50, 0], X_train[:50, 1], marker_size, c='blue', marker='o', label='train data')  # class 1
    ax.scatter(X_train[50:, 0], X_train[50:, 1], marker_size, c='red',  marker='o', label='train data')  # class -1
    
    #TODO: implement a better plot (only circles)
    # plot the support vectors
    ax.scatter(X_train[sv_indices_c1, 0], X_train[sv_indices_c1, 1], marker_size + 50, facecolors='none', edgecolors='k', marker='o', label='train data')  # class 1
    ax.scatter(X_train[sv_indices_c2, 0], X_train[sv_indices_c2, 1], marker_size + 50, facecolors='none', edgecolors='k', marker='o', label='train data')  # class 1



def plot_decision_boundary(X, w, ax):

    limits = ax.axis()
    ax.axis(limits)
    dx1 = np.array([-6, 2])
    dx2 = -((w[0] + dx1*w[1])/w[2]).flatten()
    ax.plot(dx1, dx2, '-', color="black", linewidth=2.0)

    yy = -(w[0] + dx1*w[1])/w[2]
    a = -w[0] / w[1]
    margin = 1 / np.sqrt(np.sum(w**2))
    yy_down = yy - np.sqrt(1 + a**2) * margin
    yy_up = yy + np.sqrt(1 + a**2) * margin
    ax.plot(dx1, yy_down, "grey")
    ax.plot(dx1, yy_up, "grey")

    # plot the line, the points, and the nearest vectors to the plane



def __proximal_subgradient_method(PHI, y, alpha, lambda_, w_i_1, delta):
    N = y.shape[0]
    epoch = 0
    w_i_1_schlange_old = np.zeros((2, 1))
    w_i_1_schlange = np.zeros((2,1))
    b = w_i_1[0]
    loss = []

    PHI = PHI[:, 1:] #100x2
    w_i_1 = w_i_1[1:] #2x1

    while (42):
        epoch += 1
        #w_schlange = np.concatenate((b.reshape(1, 1), w_i_1))
        w_schlange = w_i_1
        g = np.mean(np.where(y.reshape((100, 1)) * (PHI @ w_i_1) + b >= 1, 0, -y.reshape((100, 1)) * PHI), axis=0).reshape(2, 1)
        numerical_grad = 0 

        # for i in range(y.shape[0]):
        #     numerical_grad += approx_fprime(PHI[i], __hinge_loss, 1e-8, y[i], lambda_, w_schlange)

        # numerical_grad = numerical_grad / N
        
        w_i_1_schlange = w_schlange - alpha * g#  numerical_grad.reshape(3, 1)
        b = np.mean(-(PHI @ w_i_1_schlange))
        l = hinge_loss(PHI, w_i_1_schlange, y, lambda_)
        loss.append(l)
        w_i_1 = w_i_1 / (1 + lambda_ * alpha)

        # compare numerical and analytical gradients
        grad_diff = np.abs(g.flatten() - numerical_grad).flatten()

        w_diff = np.abs(w_i_1_schlange - w_i_1_schlange_old).flatten()
        # print(f'Gradient, w and b diff for iteration {epoch} = {grad_diff} | {w_diff} | {b} | {l}.')
        if epoch % 100 == 0:
            print(f'Gradient, w and b diff for iteration {epoch} = {grad_diff} | {w_diff} | {b} | {l}.')

        if epoch > 1 and np.all(w_diff < delta):
            print(f'Proximal subgradient method converged after {epoch} epochs using delta = {delta} for all coordinates of gradient vector.')
            return w_i_1_schlange, b, loss

        w_i_1_schlange_old = w_i_1_schlange




def __hinge_loss(x, y, lambda_, w_i_1):
    w = w_i_1[1:]
    b = w_i_1[0]
    phi = x[1:]
    return lambda_ / 2 * np.linalg.norm(w) ** 2 + np.maximum(0, 1 - y * (w.T @ phi + b))

def hinge_loss(PHI, w, y, l_):
    y = y.reshape((len(y), 1))
    return np.sum(l_/2 * np.linalg.norm(w) ** 2 + np.maximum(0, 1 - y * (PHI @ w)))

def create_kernel(Xn, Xm, sigma):
    """ Create kernel defined in eq (13)
    """
    K_matrix = np.exp(- np.linalg.norm(Xn.reshape(Xn.shape[0], Xn.shape[1], 1) - Xm.reshape(Xm.shape[0], Xm.shape[1], 1).T, axis=1) / (2 * sigma**2))  # np.amax(K-K_matrix) = max_diff =
    # 0.10520138377921817

    return K_matrix



def Q(y, K):
    Q_ = np.zeros_like(K)
    Q_ = (y @ y.T) * K

    return Q_


def D(alpha, Q_):
    return np.sum(alpha - 1/2 * alpha.T @ Q_ @ alpha, axis=0)


def grad_D(alpha, Q_):
    return 1 - Q_ @ alpha


def projected_gradient_ascent(N, step_size, y, K, delta):
    """ task 4: Projected gradient descent as described in algorithm 3.
    """

    a_i = np.zeros((N, 1))  # initialize Lagrange coefficients
    epochs = 0  # counter variable used for error tracking
    Q_ = Q(y, K)  # define matrix Q
    loss = []

    while True:
        epochs += 1
        dD = grad_D(a_i, Q_)
        a_i_1 = np.maximum(0, a_i + step_size * dD)
        loss.append(D(a_i_1, Q_))

        # TODO: verify gradient

        # Check convergence
        a_diff = np.sum(np.abs(a_i_1 - a_i))
        if epochs % 100 == 0:
            print(f'Iteration {epochs} has an absolute summed difference of Lagrange multipliers {a_diff}.')

        if a_diff < delta:
            print(f'Final Lagrange multiplier difference after {epochs} epochs is {a_diff}.')
            return a_i_1, loss

        a_i = a_i_1


def plot_svm_dual(X, y, y_pred, alpha, ax):

    marker_size = 60
    c1_ind = np.nonzero(y == 1)
    c2_ind = np.nonzero(y == -1)
    ax.scatter(X[c1_ind, 0], X[c1_ind, 1], marker_size, c='blue', marker='o', label='train data')  # class 1
    ax.scatter(X[c2_ind, 0], X[c2_ind, 1], marker_size, c='red', marker='o', label='train data')  # class -1
    
    c1_ind = np.argwhere(y_pred == 1)
    c2_ind = np.argwhere(y_pred == -1)
    ax.scatter(X[c1_ind, 0], X[c1_ind, 1], marker_size, c='k', marker='+', label='train data pred')  # class 1
    ax.scatter(X[c2_ind, 0], X[c2_ind, 1], marker_size, c='k', marker='x', label='train data pred')  # class -1

    ax.legend()


def decision_boundary_svm_dual(ax, X, y, K, a, sigma, plot_step=0.2):
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]

    Z = (create_kernel(grid, X, sigma) @ (a * y))

    #plotSVM(w, ax)
    Z = Z.reshape(xx.shape)

    ax.imshow(
        Z,
        interpolation="nearest",
        extent=(xx.min(), xx.max(), yy.min(), yy.max()),
        aspect="auto",
        origin="lower",
        cmap=plt.cm.PuOr_r,
    )

    decision_boundary = np.where(np.round(Z, 1) == 0, 1, 0)
    ax.contour(xx, yy, decision_boundary, colors='k')

    # class1 = np.where(np.round(Z, 2) == 1, 1, 0)
    # ax.contour(xx, yy, class1, colors='grey')

    # class2 = np.where(np.round(Z, 2) == -1, 1, 0)
    # ax.contour(xx, yy, class2, colors='grey')


def bias(alpha, yi, yj, K):
    return np.mean(yi.T - yj * (alpha.T @ K))


def svm_dual():
    """ Subtask 4: Dual SVM

        Requirements for the plot:
        - the first subplot should contain the energy of the objective function plotted over the iterations
        - the next two subplots should contain predictions of train data B and test data B, respectively, in a scatterplot
            - indicate the two predicted classes with 2 different colors for both data sets 
            - use a legend
            - for both train and test data also plot the separating hyperplane and the margin at \pm 1
            - for the train data include the support vectors
    """

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    plt.suptitle('Task 4 - Dual Support Vector Machine', fontsize=12)
    
    ax[0].set_title('Energy $D(\mathbf{a})$')
    ax[1].set_title('Train Predictions B')
    ax[2].set_title('Test Predictions B')

    """ Start of your code 
    """

    N_train = len(data_b_train)
    N_test = len(data_b_test)

    X_train = np.stack([data_b_train[:, 0], data_b_train[:, 1]]).T  # X = 100x2
    y_train = data_b_train[:, 2].reshape((N_train, 1))  # y = 100x1

    X_test = np.stack([data_b_test[:, 0], data_b_test[:, 1]]).T  # X = 100x2
    y_test = data_b_test[:, 2].reshape((N_test, 1))  # y = 100x1
    
    # hyperparameters
    sigma = 1
    step_size = 1e-3
    delta = 1e-4

    kernel_train = create_kernel(X_train, X_train, sigma)
    kernel_test = create_kernel(X_test, X_train, sigma)
    a, loss = projected_gradient_ascent(N_train, step_size, y_train, kernel_train, delta)

    y_train_pred = np.sign((kernel_train @ a) * y_train)
    y_test_pred = np.sign((kernel_test @ a) * y_test)
    train_acc, test_acc = len(np.nonzero(y_train == y_train_pred)[0]) / N_train, len(np.nonzero(y_test == y_test_pred)[0]) / N_test

    print(f'SVM dual space: train_acc = {train_acc}, test_acc = {test_acc}')

    # plots

    # TODO: calculate support vectors in dual form
    # (kernel_train @ a) * y_train

    plot_svm_dual(X_train, y_train, y_train_pred, a, ax[1])
    plot_svm_dual(X_test, y_test, y_test_pred, a, ax[2])

    #plot the loss
    ax[0].loglog(range(1, len(loss)), loss[:-1]-min(loss))
    ax[0].set_xlabel('iterations')
    ax[0].set_ylabel('$D(\mathbf{a})$')

    decision_boundary_svm_dual(ax[1], X_train, y_train, kernel_train, a, sigma)
    decision_boundary_svm_dual(ax[2], X_train, y_train, kernel_train, a, sigma)
    plt.show()

    """ End of your code
    """

    ax[1].legend()
    ax[2].legend()

    return fig


if __name__ == '__main__':
    # load train/test datasets A and B globally
    global data_a_train, data_a_test, data_b_train, data_b_test, sigmoid
    data_a_train, data_a_test, data_b_train, data_b_test = load_data()
    sigmoid = (lambda x: 1/(1+np.exp(-x)))

    tasks = [quadratic, logistic, svm_primal, svm_dual]
    pdf = PdfPages('figures.pdf')
    for task in tasks:
        f = task()
        pdf.savefig(f)

    pdf.close()
