import numpy as np    # it is an unofficial standard to use np for numpy
import time


"""
VECTOR CREATION
NumPy's basic data structure is an indexable, n-dimensional array containing elements of the same type (dtype). Right away, you may notice we have overloaded the term 'dimension'. Above, it was the number of elements in the vector, here, dimension refers to the number of indexes of an array. 
A one-dimensional or 1-D array has one index. In Course 1, we will represent vectors as NumPy 1-D arrays.
1-D array, shape (n,): n elements indexed [0] through [n-1]
Data creation routines in NumPy will generally have a first parameter which is the shape of the object. 
This can either be a single value for a 1-D result or a tuple (n,m,...) specifying the shape of the result. 
Below are examples of creating vectors using these routines.
"""

# NumPy routines which allocate memory and fill arrays with value
a = np.zeros(4);                print(f"np.zeros(4) :   a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
a = np.zeros((4,));             print(f"np.zeros(4,) :  a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
a = np.random.random_sample(4); print(f"np.random.random_sample(4): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")

#Some data creation routines do not take a shape tuple:
# NumPy routines which allocate memory and fill arrays with value but do not accept shape as input argument
a = np.arange(4.);              print(f"np.arange(4.):     a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
a = np.random.rand(4);          print(f"np.random.rand(4): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")

# values can be specified manually as well.
# NumPy routines which allocate memory and fill with user specified values
a = np.array([5,4,3,2]);  print(f"np.array([5,4,3,2]):  a = {a},     a shape = {a.shape}, a data type = {a.dtype}")
a = np.array([5.,4,3,2]); print(f"np.array([5.,4,3,2]): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")


"""
OPERATIONS IN VECTORS
INDEXING
Elements of vectors can be accessed via indexing and slicing. 
NumPy provides a very complete set of indexing and slicing capabilities. 
We will explore only the basics needed for the course here. Reference Slicing and Indexing for more details.
Indexing means referring to an element of an array by its position within the array.
Slicing means getting a subset of elements from an array based on their indices.
NumPy starts indexing at zero so the 
3rd element of an vector  𝐚  is a[2].
"""
#vector indexing operations on 1-D vectors
a = np.arange(10)
print(a)

#access an element
print(f"a[2].shape: {a[2].shape} a[2]  = {a[2]}, Accessing an element returns a scalar")

# access the last element, negative indexes count from the end
print(f"a[-1] = {a[-1]}")

#indexs must be within the range of the vector or they will produce and error
try:
    c = a[10]
except Exception as e:
    print("The error message you'll see is:")
    print(e)



"""
OPERATIONS IN VECTORS
SLICING
Slicing creates an array of indices using a set of three values (start:stop:step). 
A subset of values is also valid. Its use is best explained by example:
"""
#vector slicing operations
a = np.arange(10)
print(f"a         = {a}")

#access 5 consecutive elements (start:stop:step)
c = a[2:7:1];     print("a[2:7:1] = ", c)

# access 3 elements separated by two
c = a[2:7:2];     print("a[2:7:2] = ", c)

# access all elements index 3 and above
c = a[3:];        print("a[3:]    = ", c)

# access all elements below index 3
c = a[:3];        print("a[:3]    = ", c)

# access all elements
c = a[:];         print("a[:]     = ", c)


"""
SINGLE VECTOR OPERATIONS
There are a number of useful operations that involve operations on a single vector."""
a = np.array([1,2,3,4])
print(f"a             : {a}")
# negate elements of a
b = -a
print(f"b = -a        : {b}")

# sum all elements of a, returns a scalar
b = np.sum(a)
print(f"b = np.sum(a) : {b}")

b = np.mean(a)
print(f"b = np.mean(a): {b}")

b = a**2
print(f"b = a**2      : {b}")


""" 
VECTOR VECTOR ELEMENT-WISE OPERATIONS
Most of the NumPy arithmetic, logical and comparison operations apply to vectors as well. 
These operators work on an element-by-element basis. 
For example
𝑐𝑖=𝑎𝑖+𝑏𝑖
"""
a = np.array([ 1, 2, 3, 4])
b = np.array([-1,-2, 3, 4])
print(f"Binary operators work element wise: {a + b}")

# Of course, for this to work correctly, the vectors must be of the same size:
#try a mismatched vector operation
c = np.array([1, 2])
try:
    d = a + c
except Exception as e:
    print("The error message you'll see is:")
    print(e)


"""
SCALAR VECTOR OPERATIONS
Vectors can be 'scaled' by scalar values. A scalar value is just a number. 
The scalar multiplies all the elements of the vector.
"""
a = np.array([1, 2, 3, 4])

# multiply a by a scalar
b = 5 * a
print(f"b = 5 * a : {b}")

"""
VECTOR VECTOR DOT PRODUCT // instead of slow For loops in python
The dot product is a mainstay of Linear Algebra and NumPy. 
This is an operation used extensively in this course and should be well understood. The dot product is shown below.
The dot product multiplies the values in two vectors element-wise and then sums the result. Vector dot product requires the dimensions of the two vectors to be the same.

Let's implement our own version of the dot product below:

Using a for loop, implement a function which returns the dot product of two vectors. 
The function to return given inputs 𝑎 and 𝑏:
Assume both a and b are the same shape.
"""


def my_dot(a, b):
    """
   Compute the dot product of two vectors

    Args:
      a (ndarray (n,)):  input vector
      b (ndarray (n,)):  input vector with same dimension as a

    Returns:
      x (scalar):
    """
    x = 0
    for i in range(a.shape[0]):
        x = x + a[i] * b[i]
    return x

# test 1-D
a = np.array([1, 2, 3, 4])
b = np.array([-1, 4, 3, 2])
print(f"my_dot(a, b) = {my_dot(a, b)}")
# Note, the dot product is expected to return a scalar value.

#Let's try the same operations using np.dot.
# test 1-D
a = np.array([1, 2, 3, 4])
b = np.array([-1, 4, 3, 2])
c = np.dot(a, b)
print(f"NumPy 1-D np.dot(a, b) = {c}, np.dot(a, b).shape = {c.shape} ")
c = np.dot(b, a)
print(f"NumPy 1-D np.dot(b, a) = {c}, np.dot(a, b).shape = {c.shape} ")
# Above, you will note that the results for 1-D matched our implementation.


"""THE NEED FOR SPEED: VECTOR VS FOR LOOP
So, vectorization provides a large speed up in this example. 
This is because NumPy makes better use of available data parallelism in the underlying hardware. 
GPU's and modern CPU's implement Single Instruction, Multiple Data (SIMD) pipelines allowing multiple operations to be issued in parallel. 
This is critical in Machine Learning where the data sets are often very large."""
np.random.seed(1)
a = np.random.rand(10000000)  # very large arrays
b = np.random.rand(10000000)

tic = time.time()  # capture start time
c = np.dot(a, b)
toc = time.time()  # capture end time

print(f"np.dot(a, b) =  {c:.4f}")
print(f"Vectorized version duration: {1000*(toc-tic):.4f} ms ")

tic = time.time()  # capture start time
c = my_dot(a,b)
toc = time.time()  # capture end time

print(f"my_dot(a, b) =  {c:.4f}")
print(f"loop version duration: {1000*(toc-tic):.4f} ms ")

del(a);del(b)  #remove these big arrays from memory


