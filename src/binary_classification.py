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

def load_data():
    """ General utility function to load provided .npy data

        For datasets A and B:
        - load train/test data (N/N_t samples)
        - the first 2 columns are input x and the last column is target y for all N/N_t samples
        - include transform to homogeneous input data
    """

    data_a_train, data_a_test = None, None
    data_b_train, data_b_test = None, None

    """ Start of your code 
    """


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


    """ End of your code
    """

    ax[0].legend()
    ax[1].legend()
    return fig

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


    """ End of your code
    """

    ax[1].legend()
    ax[2].legend()
    return fig

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


    """ End of your code
    """

    ax[1].legend()
    ax[2].legend()
    return fig

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
    data_a_train, data_a_test, data_b_train, data_b_test = load_data()

    tasks = [quadratic, logistic, svm_primal, svm_dual]
    pdf = PdfPages('figures.pdf')
    for task in tasks:
        f = task()
        pdf.savefig(f)

    pdf.close()
