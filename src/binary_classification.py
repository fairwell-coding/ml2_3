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

    data_a_train, data_a_test = np.load('src/data_a_test.npy'), np.load('src/data_a_test.npy')
    data_b_train, data_b_test = np.load('src/data_b_train.npy'), np.load('src/data_b_test.npy')

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
 
    train_acc, test_acc = len(np.nonzero(Y_train == Y_train_pred)[0]) / N, len(np.nonzero(Y_test == Y_test_pred)[0]) / N

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
    w0 = np.random.uniform(size=(M, 1))

    k_max = 1000
    #TODO: calculate engery
    E = np.zeros((k_max, 1))
    w, E = nesterov_gradient(Y_train, X_train, w0, L, k_max)
    # Y_train_pred = np.sign(sigmoid(PHI @ w) - 0.5)
    Y_train_pred = np.where(sigmoid(PHI @ w) < 0.5, -1 ,1)

    X_test = np.stack([data_a_test[:, 0], data_a_test[:, 1]]).T # X = 100x2
    Y_test = data_a_test[:, 2] # Y = 100x1

    PHI = feature_transform(X_test)
    Y_test_pred = np.where(sigmoid(PHI @ w) < 0.5, -1 ,1)


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


def log_loss(X, y, w):
    N = len(y)
    PHI = X#feature_transform(X.reshape((N, 2)))
    w = w.reshape((3,1))
    y = y.reshape((N, 1))
    return np.sum(np.log(1 + np.exp(-y * PHI @ w)), axis=0)
    # return -np.sum(np.log(sigmoid(y * PHI @ w)))

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
        E_.append(log_loss(PHI, y, w_))
        if (k-1)%10 == 0:      
            E_app = approx_fprime(w_.flatten(), lambda w__: log_loss(PHI, y, w__), 1e-8)
            print(f'{log_loss(PHI, y, w_)} | {E_app.flatten()} | {grad_log_loss(PHI, y, w_).flatten()}')
        w_old = w
        w = w_ - 1/L * grad_log_loss(PHI, y, w_).reshape((3, 1))

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

    # load data
    X_train = np.stack([data_a_train[:, 0], data_a_train[:, 1]]).T  # X = 100x2
    y_train = data_a_train[:, 2]  # y = 100x1
    X_test = np.stack([data_a_test[:, 0], data_a_test[:, 1]]).T  # X = 100x2
    y_test = data_a_test[:, 2]  # y = 100x1

    # hyperparameters
    lambda_ = 1e-4  # 1.24
    alpha = 0.2
    w = np.ones((3, 1))
    w[0] = 0.0
    delta = 1e-4

    PHI = feature_transform(X_train)
    w, b = __proximal_subgradient_method(PHI, y_train, alpha, lambda_, w, delta)

    y_train_pred = np.sign(feature_transform(X_train) @ w + b)
    y_test_pred = np.sign(feature_transform(X_test) @ w + b)

    train_acc, test_acc = len(np.nonzero(y_train.reshape(y_train.shape[0], 1) == y_train_pred)[0]) / X_train.shape[0], len(np.nonzero(y_test.reshape(y_test.shape[0], 1) == y_test_pred)[0]) / \
                          X_test.shape[0]
    print(f'SVM primal space: train_acc = {train_acc}, test_acc = {test_acc}')

    __plots_svm_primal(PHI, X_train, ax, w, b)

    """ End of your code
    """

    ax[1].legend()
    ax[2].legend()
    return fig


def __plots_svm_primal(PHI, X_train, ax, w, b):
    marker_size = 30
    ax[0].scatter(X_train[:50, 0], X_train[:50, 1], marker_size, c='blue', marker='o', label='train data')  # class 1
    ax[0].scatter(X_train[50:, 0], X_train[50:, 1], marker_size, c='red', marker='o', label='train data')  # class -1
    plot_decision_boundary(X_train, w, ax[0], b, plot_step=0.01)
    plt.show()


def plot_decision_boundary(X, w, ax, b, plot_step=0.05):
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                           np.arange(y_min, y_max, plot_step))

    Z = feature_transform(np.c_[xx.ravel(), yy.ravel()]) @ w + b
    Z = Z.reshape(xx.shape)
    hyperplane = np.where(np.round(Z, 2) == 0, 1, 0)
    ax.contour(xx, yy, hyperplane, colors='black')

    class1 = np.where(np.round(Z, 2) == 1, 1, 0)
    ax.contour(xx, yy, class1, colors='grey')

    class2 = np.where(np.round(Z, 2) == -1, 1, 0)
    ax.contour(xx, yy, class2, colors='grey')


def __proximal_subgradient_method(PHI, y, alpha, lambda_, w_i_1, delta):
    N = y.shape[0]
    epoch = 0
    w_i_1_schlange_old = np.zeros((3, 1))
    b = w_i_1[0]

    while (42):
        b = np.mean(y.reshape(100, 1) - (PHI @ w_i_1))
        w_schlange = np.concatenate((b.reshape(1, 1), w_i_1[1:]))
        # g = np.where(y.T @ (PHI @ w_schlange) >= 1, np.zeros((3, )), - PHI.T @ y).reshape(3, 1) * 1 / N  # analytical_grad = g
        g = np.mean(np.where(y.reshape((100, 1)) * (PHI @ w_i_1 + b) >= 1, 0, -y.reshape((100, 1)) * PHI), axis=0).reshape(3, 1)
        numerical_grad = 0
        for i in range(y.shape[0]):
            numerical_grad += approx_fprime(PHI[i], __hinge_loss, 1e-8, y[i], lambda_, w_schlange)

        numerical_grad = numerical_grad / N
        # w_i_1_schlange = w_schlange - alpha * g
        # w_i_1 = w_i_1 / (1 + lambda_ * alpha)
        
        w_i_1_schlange = w_schlange - alpha * numerical_grad.reshape(3, 1)
        w_i_1 = w_i_1 / (1 + lambda_ * alpha)

        # compare numerical and analytical gradients


        grad_diff = np.abs(g.flatten() - numerical_grad).flatten()

        w_diff = np.abs(w_i_1_schlange - w_i_1_schlange_old).flatten()
        print(f'Gradient, w and b diff for iteration {epoch} = {grad_diff} | {w_diff} | {b}.')

        if epoch > 1000 or np.all(w_diff < 1e-8):
            return w_i_1_schlange, b

        w_i_1_schlange_old = w_i_1_schlange

        epoch += 1

    print(f'Proximal subgradient method converged after {epoch} epochs using delta = {delta} for all coordinates of gradient vector.')


# def __hinge_loss(phi, y, lambda_, w_i_1):
#     return (lambda_ / 2) * np.linalg.norm(w_i_1[1:]) ** 2 + np.maximum(0, 1 - y * (w_i_1.T @ phi)).reshape(1)


def __hinge_loss(x, y, lambda_, w_i_1):
    # return lambda / 2 * np.linalg.norm(w_i_1) ** 2 + 1 / N * np.sum(np.maximum(0, 1 - y_train.T @ PHI @ w_i_1), axis=0)
    w = w_i_1[1:]
    b = w_i_1[0]
    phi = x[1:]
    return lambda_ / 2 * np.linalg.norm(w) ** 2 + np.maximum(0, 1 - y * (w.T @ phi + b))


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
    tasks = [quadratic]
    pdf = PdfPages('figures.pdf')
    for task in tasks:
        f = task()
        pdf.savefig(f)

    pdf.close()

