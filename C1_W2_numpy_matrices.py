"""bstract
Matrices, are two dimensional arrays. The elements of a matrix are all of the same type.
In notation, matrices are denoted with capitol, bold letter such as  𝐗
 . In this and other labs, m is often the number of rows and n the number of columns.
 The elements of a matrix can be referenced with a two dimensional index.
 In math settings, numbers in the index typicallyrun from 1 to n.
 In computer science and these labs, indexing will run from 0 to n-1."""

"""NumPy Arrays
NumPy's basic data structure is an indexable, n-dimensional array containing elements of the same type (dtype). These were described earlier. Matrices have a two-dimensional (2-D) index [m,n].

In Course 1, 2-D matrices are used to hold training data. Training data is  𝑚  examples by  𝑛 features creating an (m,n) array. 
Course 1 does not do operations directly on matrices but typically extracts an example as a vector and operates on that. 
Below you will review:
    - data creation
    - slicing and indexing"""

""" MATRIX CREATION 
The same functions that created 1-D vectors will create 2-D or n-D arrays. Here are some examples
Below, the shape tuple is provided to achieve a 2-D result. Notice how NumPy uses brackets to denote each dimension. 
Notice further than NumPy, when printing, will print one row per line.
"""
a = np.zeros((1, 5))
print(f"a shape = {a.shape}, a = {a}")

a = np.zeros((2, 1))
print(f"a shape = {a.shape}, a = {a}")

a = np.random.random_sample((1, 1))
print(f"a shape = {a.shape}, a = {a}")

#One can also manually specify data. Dimensions are specified with additional brackets matching the format in the printing above.
# NumPy routines which allocate memory and fill with user specified values
a = np.array([[5], [4], [3]]);   print(f" a shape = {a.shape}, np.array: a = {a}")
a = np.array([[5],   # One can also
              [4],   # separate values
              [3]]); #into separate rows
print(f" a shape = {a.shape}, np.array: a = {a}")

"""OPERATIONS ON MATRICES: INDEXING
Matrices include a second index. The two indexes describe [row, column]. 
Access can either return an element or a row/column. See below:
worth drawing attention to the last example. Accessing a matrix by just specifying the row will return a 1-D vector.
"""
#vector indexing operations on matrices
a = np.arange(6).reshape(-1, 2)   #reshape is a convenient way to create matrices
print(f"a.shape: {a.shape}, \na= {a}")

#access an element
print(f"\na[2,0].shape:   {a[2, 0].shape}, a[2,0] = {a[2, 0]},     type(a[2,0]) = {type(a[2, 0])} Accessing an element returns a scalar\n")

#access a row
print(f"a[2].shape:   {a[2].shape}, a[2]   = {a[2]}, type(a[2])   = {type(a[2])}")
""" Reshape
The previous example used reshape to shape the array.
a = np.arange(6).reshape(-1, 2)
This line of code first created a 1-D Vector of six elements. It then reshaped that vector into a 2-D array using the reshape command. This could have been written:
a = np.arange(6).reshape(3, 2)
To arrive at the same 3 row, 2 column array. The -1 argument tells the routine to compute the number of rows given the size of the array and the number of columns.
"""

""" OPERATIONS ON MATRICES: SLICING
Slicing is an important way of accessing subarrays. The syntax is straightforward: a[start:stop:step]. 
The start position is included but the stop position is not included.
"""

#vector 2-D slicing operations
a = np.arange(20).reshape(-1, 10)
print(f"a = \n{a}")

#access 5 consecutive elements (start:stop:step)
print("a[0, 2:7:1] = ", a[0, 2:7:1], ",  a[0, 2:7:1].shape =", a[0, 2:7:1].shape, "a 1-D array")

#access 5 consecutive elements (start:stop:step) in two rows
print("a[:, 2:7:1] = \n", a[:, 2:7:1], ",  a[:, 2:7:1].shape =", a[:, 2:7:1].shape, "a 2-D array")

# access all elements
print("a[:,:] = \n", a[:,:], ",  a[:,:].shape =", a[:,:].shape)

# access all elements in one row (very common usage)
print("a[1,:] = ", a[1,:], ",  a[1,:].shape =", a[1,:].shape, "a 1-D array")
# same as
print("a[1]   = ", a[1],   ",  a[1].shape   =", a[1].shape, "a 1-D array")
