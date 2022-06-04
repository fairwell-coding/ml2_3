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

    N = len(Y_train)
    M = 1 +  X_train.shape[1]

    # calculate weight vector w
    PHI = feature_transform(X_train)
    w = invert(PHI.T@PHI) @ PHI.T @ Y_train
    Y_train_pred = np.sign(PHI @ w.T)
    # w1 = __calculate_w(PHI, Y)
    # w - w1 = [0.0, 0.0, 7.632783294297951e-17]

    X_test = np.stack([data_a_test[:, 0], data_a_test[:, 1]]).T # X = 100x2
    Y_test = data_a_test[:, 2] # Y = 100x1
    PHI =  feature_transform(X_test)
    Y_test_pred = np.sign(PHI @ w.T)

    train_error, test_error = len(np.nonzero(Y_train == Y_train_pred)[0]) / N, len(np.nonzero(Y_test == Y_test_pred)[0]) / N


    """ End of your code
    """

    ax[0].legend()
    ax[1].legend()
    plt.show()
    return fig

def feature_transform(X):
    N = X.shape[0]
    return np.stack([np.ones((N,)), X[:, 0], X[:, 1]**2]).T # PHI = 100x3

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
    w0 = np.random.uniform(size=(M, 1))

    k_max = 1000
    #TODO: calculate engery
    E = np.zeros((k_max, 1))
    E_app = np.zeros((k_max, 1))
    w, E, E_app = nesterov_gradient(Y_train, PHI, w0, L, k_max)
    Y_train_pred = np.sign(sigmoid(PHI @ w) - 0.5)

    X_test = np.stack([data_a_test[:, 0], data_a_test[:, 1]]).T # X = 100x2
    Y_test = data_a_test[:, 2] # Y = 100x1

    PHI = feature_transform(X_test)
    L = calculate_lipschitz(PHI)
    w0 = np.random.uniform(size=(M, 1))
    Y_test_pred = np.sign(sigmoid(PHI @ w) - 0.5)


    ax[0].plot(np.arange(1, k_max+1), E)

    marker_size = 30
    ax[1].scatter(X_train[:, 0], X_train[:, 1], marker_size*Y_train, c='blue', marker='o', label='train data')
    ax[1].scatter(X_train[:, 0], X_train[:, 1], marker_size*Y_train_pred, c='red',  marker='x', label='train prediction')

    ax[2].scatter(X_test[:, 0], X_test[:, 1], marker_size*Y_test, c='k', marker='o', label='test data')
    ax[2].scatter(X_test[:, 0], X_test[:, 1], marker_size*Y_test_pred, c='orange', marker='x', label='test prediction')

    """ End of your code
    """

    ax[1].legend()
    ax[2].legend()
    plt.show()
    return fig

def calculate_lipschitz(PHI):
    return 1/4 * np.amax(sigmoid(PHI)) #largest singular value

def nesterov_gradient(y, PHI, w_k, L, k_max):

    E_ = np.zeros((k_max, 1))
    E_app = np.zeros((k_max, 1))
    E_prime = lambda w_: sigmoid(y @ PHI @ w_) - 1
    E = lambda w_: -np.log(sigmoid(y @ PHI @ w_))
    w_prev = w_k
    for k in range(1, k_max+1):
        beta = (k-1)/(k+1)
        w_k = w_k + beta * (w_k - w_prev)
        w_prev = w_k

        #TODO: calculate energy (E), not correct yet
        E_[k-1] = E_prime(w_k)
        # E_app[k-1] = approx_fprime(w_k, E)
        w_k = w_k - 1/L * E_prime(w_k)

    return w_k, E_, E_app

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

    # load data
    X_train = np.stack([data_a_train[:, 0], data_a_train[:, 1]]).T # X = 100x2
    y_train = data_a_train[:, 2] # Y = 100x1

    # hyperparameters
    lambda_ = 1
    alpha = 1e-3
    w = np.zeros((3, 1))
    delta = 1e-4

    PHI = feature_transform(X_train)
    w = __proximal_subgradient_method(X_train, y_train, PHI, alpha, lambda_, w, delta)
    hyperplane = PHI @ w

    marker_size = 30
    ax[0].scatter(X_train[:50, 0], X_train[:50, 1], marker_size, c='blue', marker='o', label='train data')  # class 1
    ax[0].scatter(X_train[50:, 0], X_train[50:, 1], marker_size, c='red', marker='o', label='train data')  # class -1
    # ax[1].scatter(X_train[:, 0], X_train[:, 1], marker_size*y_train_pred, c='red',  marker='x', label='train prediction')
    plt.show()

    """ End of your code
    """

    ax[1].legend()
    ax[2].legend()
    return fig


def __proximal_subgradient_method(X_train, y_train, PHI, alpha, lambda_, w_i, delta):
    epoch = 0

    while (42):
        g = np.where(y_train.T @ PHI @ w_i >= 1, 0, - PHI.T @ y_train)
        w_i_1 = w_i - alpha * g.reshape(3,1)
        w_i_1 = w_i_1 / (1 + lambda_ * alpha)

        grad_diff = np.abs(w_i_1 - w_i)
        print(f'Gradient diff for iteration {epoch} = {grad_diff}.')

        if np.all(grad_diff < delta):
            return w_i_1

        w_i = w_i_1
        epoch += 1


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

    fig, ax = plt.subplots(1,3, figsize=(15,5))
    plt.suptitle('Task 4 - Dual Support Vector Machine', fontsize=12)
    
    ax[0].set_title('Energy $D(\mathbf{a})$')
    ax[1].set_title('Train Predictions B')
    ax[2].set_title('Test Predictions B')

    """ Start of your code 
    """


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

    # tasks = [quadratic, logistic, svm_primal, svm_dual]
    tasks = [svm_primal]
    pdf = PdfPages('figures.pdf')
    for task in tasks:
        f = task()
        pdf.savefig(f)

    pdf.close()
