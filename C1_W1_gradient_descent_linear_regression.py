import math, copy
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')
from lab_utils_uni import plt_house_x, plt_contour_wgrad, plt_divergence, plt_gradients

"""
implement Gradient Descent
You will implement gradient descent algorithm for one feature. You will need three functions.

compute_gradient implementing equation (4) and (5) above
compute_cost implementing equation (2) above (code from previous lab)
gradient_descent, utilizing compute_gradient and compute_cost
Conventions:

The naming of python variables containing partial derivatives follows this pattern, 
∂𝐽(𝑤,𝑏)/∂𝑏  will be dj_db.
w.r.t is With Respect To, as in partial derivative of  𝐽(𝑤𝑏)   With Respect To  𝑏

"""

# Load our data set
x_train = np.array([1.0, 2.0])   #features
y_train = np.array([300.0, 500.0])   #target value


# Function to calculate the cost
def compute_cost(x, y, w, b):
    m = x.shape[0]
    cost = 0

    for i in range(m):
        f_wb = w * x[i] + b    # calculate f_wb | prediction | y-hat
        cost = cost + (f_wb - y[i]) ** 2 #add up squared errors in a loop
    total_cost = 1 / (2 * m) * cost   # calculate total_cost adjusted

    return total_cost


# compute_gradient implements (4) and (5) above and returns
#   ∂𝐽(𝑤,𝑏)/∂𝑤
#   ∂𝐽(𝑤,𝑏)/∂𝑏

def compute_gradient(x, y, w, b):
    """
    Computes the gradient for linear regression
    Args:
      x (ndarray (m,)): Data, m examples
      y (ndarray (m,)): target values
      w,b (scalar)    : model parameters
    Returns
      dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
      dj_db (scalar): The gradient of the cost w.r.t. the parameter b
     """

    # Number of training examples
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw_i = (f_wb - y[i]) * x[i]
        dj_db_i = f_wb - y[i]
        dj_db += dj_db_i
        dj_dw += dj_dw_i
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db

plt_gradients(x_train,y_train, compute_cost, compute_gradient)  # Gradient show in quiver plot
plt.show()



def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function):
    """
    Performs gradient descent to fit w,b. Updates w,b by taking
    num_iters gradient steps with learning rate alpha

    Args:
      x (ndarray (m,))  : Data, m examples
      y (ndarray (m,))  : target values
      w_in,b_in (scalar): initial values of model parameters
      alpha (float):     Learning rate
      num_iters (int):   number of iterations to run gradient descent
      cost_function:     function to call to produce cost
      gradient_function: function to call to produce gradient

    Returns:
      w (scalar): Updated value of parameter after running gradient descent
      b (scalar): Updated value of parameter after running gradient descent
      J_history (List): History of cost values
      p_history (list): History of parameters [w,b]
      """

    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    p_history = []
    b = b_in
    w = w_in

    for i in range(num_iters):
        # Calculate the gradient and update the parameters using gradient_function
        dj_dw, dj_db = gradient_function(x, y, w, b)

        # Update Parameters using equation (3) above
        b = b - alpha * dj_db
        w = w - alpha * dj_dw

        # Save cost J at each iteration
        if i < 100000:  # prevent resource exhaustion
            J_history.append(cost_function(x, y, w, b))
            p_history.append([w, b])
        # Print cost every at intervals 10 times or as many iterations if < 10
        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ",
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")

    return w, b, J_history, p_history  # return w and J,w history for graphing


# initialize parameters
w_init = 0
b_init = 0
# some gradient descent settings
iterations = 10000
tmp_alpha = 1.0e-2
# run gradient descent
w_final, b_final, J_hist, p_hist = gradient_descent(x_train ,y_train, w_init, b_init, tmp_alpha,
                                                    iterations, compute_cost, compute_gradient)
print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})")


## Plot Cost vs iterations of GD
"""
A plot of cost versus iterations is a useful measure of progress in gradient descent. 
Cost should always decrease in successful runs. The change in cost is so rapid initially, 
]it is useful to plot the initial decent on a different scale than the final descent. 
In the plots below, note the scale of cost on the axes and the iteration step.

# plot cost versus iteration  
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12,4))
ax1.plot(J_hist[:100])
ax2.plot(1000 + np.arange(len(J_hist[1000:])), J_hist[1000:])
ax1.set_title("Cost vs. iteration(start)");  ax2.set_title("Cost vs. iteration (end)")
ax1.set_ylabel('Cost')            ;  ax2.set_ylabel('Cost') 
ax1.set_xlabel('iteration step')  ;  ax2.set_xlabel('iteration step') 
plt.show()
"""

# plot cost versus iteration
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12,4))
ax1.plot(J_hist[:100])
ax2.plot(1000 + np.arange(len(J_hist[1000:])), J_hist[1000:])
ax1.set_title("Cost vs. iteration(start)");  ax2.set_title("Cost vs. iteration (end)")
ax1.set_ylabel('Cost')            ;  ax2.set_ylabel('Cost')
ax1.set_xlabel('iteration step')  ;  ax2.set_xlabel('iteration step')
plt.show()

#Predicrtions
print(f"1000 sqft house prediction {w_final*1.0 + b_final:0.1f} Thousand dollars")
print(f"1200 sqft house prediction {w_final*1.2 + b_final:0.1f} Thousand dollars")
print(f"2000 sqft house prediction {w_final*2.0 + b_final:0.1f} Thousand dollars")

# plot during GD iterations
fig, ax = plt.subplots(1,1, figsize=(12, 6))
plt_contour_wgrad(x_train, y_train, p_hist, ax)
"""Above, the contour plot shows the  𝑐𝑜𝑠𝑡(𝑤,𝑏)
  over a range of  𝑤 and  𝑏.  Cost levels are represented by the rings. 
 Overlayed, using red arrows, is the path of gradient descent. Here are some things to note:
The path makes steady (monotonic) progress toward its goal.
initial steps are much larger than the steps near the goal."""

# Zooming in, we can see that final steps of gradient descent. Note the distance between steps shrinks as the gradient approaches zero.
fig, ax = plt.subplots(1,1, figsize=(12, 4))
plt_contour_wgrad(x_train, y_train, p_hist, ax, w_range=[180, 220, 0.5], b_range=[80, 120, 0.5],
             contours=[1,5,10,20],resolution=0.5)
fig, ax = plt.subplots(1,1, figsize=(12, 4))


""" Increasing Learning rate alpha
You will see that 𝑤 and  𝑏 are bouncing back and forth between positive and negative 
with the absolute value increasing with each iteration. 
  Further, each iteration  ∂𝐽(𝑤,𝑏)/∂𝑤  changes sign and cost is increasing rather than decreasing. 
This is a clear sign that the learning rate is too large and the solution is diverging. Let's visualize this with a plot."""
# initialize parameters
w_init = 0
b_init = 0
# set alpha to a large value
iterations = 10
tmp_alpha = 8.0e-1
# run gradient descent
w_final, b_final, J_hist, p_hist = gradient_descent(x_train ,y_train, w_init, b_init, tmp_alpha,
                                                    iterations, compute_cost, compute_gradient)
plt_divergence(p_hist, J_hist,x_train, y_train)
plt.show()


