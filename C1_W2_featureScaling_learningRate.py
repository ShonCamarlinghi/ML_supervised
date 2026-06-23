import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.preprocessing
from lab_utils_multi import load_house_data, run_gradient_descent
from lab_utils_multi import norm_plot, plt_equal_scale, plot_cost_i_w
from lab_utils_common import dlc
np.set_printoptions(precision=2)
plt.style.use('./deeplearning.mplstyle')
"""
Problem Statement
As in the previous labs, you will use the motivating example of housing price prediction. 
The training data set contains 
many examples with 4 features (size, bedrooms, floors and age) shown in the table below. 
Note, in this lab, the Size feature is in sqft while earlier labs utilized 1000 sqft. 
This data set is larger than the previous lab.

We would like to build a linear regression model using these values so we can then predict the price 
for other houses - say, a house with 1200 sqft, 3 bedrooms, 1 floor, 40 years old.
"""


# load the dataset
X_train, y_train = load_house_data()
X_features = ['size(sqft)','bedrooms','floors','age']

fig,ax=plt.subplots(1, 4, figsize=(12, 3), sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_train[:,i],y_train)
    ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel("Price (1000's)")
plt.show()

"""
The lectures discussed some of the issues related to setting the learning rate  𝛼
 . The learning rate controls the size of the update to the parameters. See equation (1) above.
  It is shared by all the parameters.
Let's run gradient descent and try a few settings of  𝛼
  on our data set
"""

#set alpha to 9.9e-7
_, _, hist = run_gradient_descent(X_train, y_train, 10, alpha = 9.9e-7)
print(hist)
plot_cost_i_w(X_train, y_train, hist)

""" It appears the learning rate is too high. The solution does not converge. 
Cost is increasing rather than decreasing. Let's plot the result:
The plot on the right shows the value of one of the parameters,  𝑤0
 . At each iteration, it is overshooting the optimal value and as a result, cost ends up increasing rather than approaching the minimum. Note that this is not a completely accurate picture as there are 4 parameters being modified each pass rather than just one. This plot is only showing  𝑤0
  with the other parameters fixed at benign values. In this and later plots you may notice the blue and orange lines being slightly off.

𝛼
  = 9e-7¶
Let's try a bit smaller value and see what happens.
"""
#set alpha to 9e-7
_,_,hist = run_gradient_descent(X_train, y_train, 10, alpha = 9e-7)
plot_cost_i_w(X_train, y_train, hist)

""" On the left, you see that cost is decreasing as it should. On the right, you can see that  𝑤0
  is still oscillating around the minimum, but the cost is decreasing with every iteration rather than increasing. 
  Note above that dj_dw[0] changes sign with each iteration as w[0] jumps over the optimal value.
   This alpha value will converge. You can vary the number of iterations to see how it behaves.
   Lets try smaller values for alpha: 𝛼  = 1e-7
"""
#set alpha to 1e-7
_,_,hist = run_gradient_descent(X_train, y_train, 10, alpha = 1e-7)
plot_cost_i_w(X_train, y_train, hist)
""" Cost is decreasing throughout the run showing that  𝛼  is not too large.
On the left, you see that cost is decreasing as it should. On the right, you can see that  𝑤0
  is approaching the minimum without oscillations. dj_w0 is negative throughout the run. 
This solution will also converge. """

"""
Andrew Ng described the importance of rescaling the dataset so the features have a similar range. If you are interested in the details of why this is the case, click on the 'details' header below. If not, the section below will walk through an implementation of how to do feature scaling.
Details
The lectures discussed three different techniques:

- Feature scaling, essentially dividing each positive feature by its maximum value, or more generally, rescale each feature by both its minimum and maximum values using (x-min)/(max-min). Both ways normalizes features to the range of -1 and 1, where the former method works for positive features which is simple and serves well for the lecture's example, and the latter method works for any features.

- Mean normalization:  𝑥𝑖:=(𝑥𝑖−𝜇𝑖)/(𝑚𝑎𝑥−𝑚𝑖𝑛)
 
- Z-score normalization which we will explore below.
"""


def zscore_normalize_features(X):
    """
    computes  X, zcore normalized by column

    Args:
      X (ndarray (m,n))     : input data, m examples, n features

    Returns:
      X_norm (ndarray (m,n)): input normalized by column
      mu (ndarray (n,))     : mean of each feature
      sigma (ndarray (n,))  : standard deviation of each feature
    """
    # find the mean of each column/feature
    mu = np.mean ( X, axis=0 )  # mu will have shape (n,)
    # find the standard deviation of each column/feature
    sigma = np.std ( X, axis=0 )  # sigma will have shape (n,)
    # element-wise, subtract mu for that column from each example, divide by std for that column
    X_norm = (X - mu) / sigma

    return (X_norm, mu, sigma)

# check our work
# from sklearn.preprocessing import scale
# scale(X_orig, axis=0, with_mean=True, with_std=True, copy=True)

#Let's look at the steps involved in Z-score normalization. The plot below shows the transformation step by step.
mu     = np.mean(X_train,axis=0)
sigma  = np.std(X_train,axis=0)
X_mean = (X_train - mu)
X_norm = (X_train - mu)/sigma

fig,ax=plt.subplots(1, 3, figsize=(12, 3))
ax[0].scatter(X_train[:,0], X_train[:,3])
ax[0].set_xlabel(X_features[0]); ax[0].set_ylabel(X_features[3]);
ax[0].set_title("unnormalized")
ax[0].axis('equal')

ax[1].scatter(X_mean[:,0], X_mean[:,3])
ax[1].set_xlabel(X_features[0]); ax[0].set_ylabel(X_features[3]);
ax[1].set_title(r"X - $\mu$")
ax[1].axis('equal')

ax[2].scatter(X_norm[:,0], X_norm[:,3])
ax[2].set_xlabel(X_features[0]); ax[0].set_ylabel(X_features[3]);
ax[2].set_title(r"Z-score normalized")
ax[2].axis('equal')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.suptitle("distribution of features before, during, after normalization")
plt.show()

"""
The plot above shows the relationship between two of the training set parameters, "age" and "size(sqft)". These are plotted with equal scale.

Left: Unnormalized: The range of values or the variance of the 'size(sqft)' feature is much larger than that of age
Middle: The first step removes the mean or average value from each feature. This leaves features that are centered around zero. It's difficult to see the difference for the 'age' feature, but 'size(sqft)' is clearly around zero.
Right: The second step divides by the standard deviation. This leaves both features centered at zero with a similar scale.
"""

## Lets normalize data and compare to original.
# normalize the original features
X_norm, X_mu, X_sigma = zscore_normalize_features(X_train)
print(f"X_mu = {X_mu}, \nX_sigma = {X_sigma}")
print(f"Peak to Peak range by column in Raw        X:{np.ptp(X_train,axis=0)}")
print(f"Peak to Peak range by column in Normalized X:{np.ptp(X_norm,axis=0)}")
#The peak to peak range of each column is reduced from a factor of thousands to a factor of 2-3 by normalization.
fig,ax=plt.subplots(1, 4, figsize=(12, 3))
for i in range(len(ax)):
    norm_plot(ax[i],X_train[:,i],)
    ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel("count");
fig.suptitle("distribution of features before normalization")
plt.show()
fig,ax=plt.subplots(1,4,figsize=(12,3))
for i in range(len(ax)):
    norm_plot(ax[i],X_norm[:,i],)
    ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel("count");
fig.suptitle("distribution of features after normalization")

plt.show()
"""
Notice, above, the range of the normalized data (x-axis) is centered around zero and roughly +/- 2. Most importantly, the range is similar for each feature.

Let's re-run our gradient descent algorithm with normalized data. Note the vastly larger value of alpha. This will speed up gradient descent.
"""
w_norm, b_norm, hist = run_gradient_descent(X_norm, y_train, 1000, 1.0e-1, )

#predict target using normalized features
m = X_norm.shape[0]
yp = np.zeros(m)
for i in range(m):
    yp[i] = np.dot(X_norm[i], w_norm) + b_norm

    # plot predictions and targets versus original features
fig,ax=plt.subplots(1,4,figsize=(12, 3),sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_train[:,i],y_train, label = 'target')
    ax[i].set_xlabel(X_features[i])
    ax[i].scatter(X_train[:,i],yp,color=dlc["dlorange"], label = 'predict')
ax[0].set_ylabel("Price"); ax[0].legend();
fig.suptitle("target versus prediction using z-score normalized model")
plt.show()

"""
The scaled features get very accurate results much, much faster!. Notice the gradient of each parameter is tiny by the end of this fairly short run. 
A learning rate of 0.1 is a good start for regression with normalized features. Let's plot our predictions versus the target values.
 Note, the prediction is made using the normalized feature while the plot is shown using the original feature values.
The results look good. A few points to note:

with multiple features, we can no longer have a single plot showing results versus features.
when generating the plot, the normalized features were used. Any predictions using the parameters learned from a normalized training set must also be normalized.
Prediction The point of generating our model is to use it to predict housing prices that are not in the data set. Let's predict the price of a house with 1200 sqft, 3 bedrooms, 1 floor, 40 years old. 
Recall, that you must normalize the data with the mean and standard deviation derived when the training data was normalized.
"""
# First, normalize out example.
x_house = np.array([1200, 3, 1, 40])
x_house_norm = (x_house - X_mu) / X_sigma
print(x_house_norm)
x_house_predict = np.dot(x_house_norm, w_norm) + b_norm
print(f" predicted price of a house with 1200 sqft, 3 bedrooms, 1 floor, 40 years old = ${x_house_predict*1000:0.0f}")

"""
COST CONTOURS
Another way to view feature scaling is in terms of the cost contours. When feature scales do not match, the plot of cost versus parameters in a contour plot is asymmetric.

In the plot below, the scale of the parameters is matched. The left plot is the cost contour plot of w[0], the square feet versus w[1],
the number of bedrooms before normalizing the features. 
The plot is so asymmetric, the curves completing the contours are not visible. In contrast, when the features are normalized,
 the cost contour is much more symmetric. The result is that updates to parameters during gradient descent can make equal progress for each parameter.
"""
plt_equal_scale(X_train, X_norm, y_train)
