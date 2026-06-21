import numpy as np
#%matplotlib widget
import matplotlib.pyplot as plt
from lab_utils_uni import plt_intuition, plt_stationary, plt_update_onclick, soup_bowl
plt.style.use('./deeplearning.mplstyle')

x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])

"""
The code below calculates cost by looping over each example. In each loop:

f_wb, a prediction is calculated
the difference between the target and the prediction is calculated and squared.
this is added to the total cost.
"""


def compute_cost(x, y, w, b):
    """
    Computes the cost function for linear regression J(w,b)
    Ref: C1_W1_Lab03_Cost_function_Soln.ipynb
    Args:
      x (ndarray (m,)): Data, m examples
      y (ndarray (m,)): target values
      w,b (scalar)    : model parameters

    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
               to fit the data points in x and y
    """
    # number of training examples
    # in lecture summation ranges are typically from 1 to m, while code will be from 0 to m-1.
    m = x.shape[0]

    cost_sum = 0
    for i in range(m):
        f_wb = w * x[i] + b   # f_wb: a prediction is calculated
        cost = (f_wb - y[i]) ** 2
        cost_sum = cost_sum + cost
    total_cost = (1 / (2 * m)) * cost_sum

    return total_cost

# Cost function intuition
plt_intuition(x_train, y_train)
"""
Notes: The plot contains a few points that are worth mentioning.
cost is minimized when  𝑤=200
 , which matches results from the previous lab
Because the difference between the target and pediction is squared in the cost equation, the cost increases rapidly when  𝑤
  is either too large or too small.
Using the w and b selected by minimizing cost results in a line which is a perfect fit to the data.
"""


# Larger dataset
x_train = np.array([1.0, 1.7, 2.0, 2.5, 3.0, 3.2])
y_train = np.array([250, 300, 480,  430,   630, 730,])
plt.close('all')
fig, ax, dyn_items = plt_stationary(x_train, y_train)
updater = plt_update_onclick(fig, ax, x_train, y_train, dyn_items)

soup_bowl()

