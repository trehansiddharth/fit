# fit

fit is a python library for fitting shapes like lines, circles, and polygons to data points. I wrote this library to help with a machine vision problem of identifying many circular objects of arbitrary scale in an image. To fit a single shape to data points I use well-known algorithms, but to identify best fits for multiple shapes (an NP-hard problem in most cases), I use a stochastic approximation algorithm that I wrote.

## Documentation

### `class Shape`
Contains methods for fitting particular shapes to data. This class is not used directly. Other classes, such as `Circle` and `Line`, inherit from `Shape` and are used for fitting particular kinds of shapes to data with specified algorithms.

The following functions can be used on any `Shape`:

* **`Shape.single(self, points)`** Fits a single copy of the shape to the data.
* **`Shape.many(self, points, numShapes)`** Fits multiple copies of the shape to the data.

To inherit from `Shape` (e.g. when implementing your own kind of shape fitter), you must implement the following functions:

* **`Shape._single(self, points)`** An algorithm for fitting a single copy of the particular shape to the data defined by `numpy.matrix` `points` (where the columns are each individual point and rows contain their coordinates). Returns a `numpy.matrix` column vector of parameters that define the fitted shape.
* **`Shape.get(self, theta)`** Takes a `numpy.matrix` column vector `theta` of parameters that define the shape and returns the parameters in a more programmer-friendly format (e.g. a tuple).
* **`Shape.getMany(self, thetas)`** Takes a `numpy.matrix` of column vectors `thetas` of parameters that defines all the shapes that fit some data, and returns them in a more programmer-friendly format (e.g. a tuple of lists).
* **`Shape.error_function(self, p, theta)`** Some sort of measure of error that defines how well a particular point `p` fits on the shape defined by parameter vector `theta`.
* **`Shape.initial_theta_function(self, points, numShapes)`** A function that stochastically generates a `numpy.matrix` of parameter column vectors that serve as the initial `theta` values in the algorithm to fit multiple shapes to data.
* **`Shape.initial_lambda_function(self, points, numShapes)`** A function that computes the best value to initialize lambda with, which is often computed based on properties such as how spread apart the points are.

### `class Circle`

Class for fitting circles to data in a scale- and translation-invariant way.

#### `circle = Circle.__init__(algorithm='bullock', solve='exact')`
Creates a circle fitter that fits a circle to a set of points. Radius and center of the circle are allowed to vary, and the algorithm for fitting a single circle and the strategy for doing so can be chosen.

* **`algorithm` (string)** Algorithm used to compute the best fit circle
    * **`'algebraic'`** Calculates a solution that minimizes the regression error, where points are defined by the algebraic equation of a circle, a*x^T*x + b^T*x + c = 0
    * **`'bullock'`** Calculates a solution that minimizes the geometric distance, defined by sum_i(g(u_i)^2), where g(u_i) = ||u_i - x||^2 - r^2
* **`solve` (string)** Technique used to find the optimum defined by the algorithm
    * **`'exact'`** Computes the exact solution using linear regression
    * **`'approximate'`** Computes an approximate solution using gradient descent

* Raises a `ValueError` if the `algorithm` or the `solve` parameters are not one of the ones listed above

#### `x, r = Circle.single(self, points)`

##### Parameters:
* **`self` (Circle)** Circle with the `algorithm` and `solve` parameters defined
* **`points` (dxn np.matrix)** A matrix of column vectors containing the coordinates of each point to fit

##### Outputs:
* **`x` (dx1 np.matrix)** Coordinates of the center of the circle
* **`r` (float)** Radius of the circle

##### Failure:
* Raises a `NotImplementedError` is solve is `'approximate'` -- gradient descent solutions haven't been implemented yet
* Running `singleCircle` on an empty matrix of points or fewer points than required to define a circle in the given dimensionality results in undefined behavior

#### `xs, rs = Circle.many(self, points, numCircles)`

##### Parameters:
* **`self` (Circle)** Circle with the `algorithm` and `solve` parameters defined
* **`points` (dxn np.matrix)** A matrix of column vectors containing the coordinates of each point to fit
* **`numCircles` (int)** Number of circles to fit to the points

##### Outputs:
* **`xs` (dxk np.matrix)** Column vectors containing the coordinates of the centers of each fitted circle
* **`rs` (1xk np.matrix)** Radius of each fitted circle

##### Failure:
* Check the failure conditions for `Circle.single`
