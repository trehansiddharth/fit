fit
=====

fit is a python library for fitting shapes like lines, circles, and polygons to data points. I wrote this library to help with a machine vision problem of identifying many circular objects of arbitary scale in an image. To fit a single shape to data points I use well-known algorithms, but to identify best fits for multiple shapes (an NP-hard problem in most cases), I use a stochastic approximation algorithm that I wrote.

Documentation
-------------

### `x, r = fit.singleCircle(points, algorithm='bullock', solve='exact')`
Fits a single circle to a set of points. Radius and center of the circle are allowed to vary.

#### Parameters:
* **`points` (dxn np.matrix)** a matrix of column vectors containing the coordinates of each point to fit
* **`algorithm` (string)** algorithm used to compute the best fit circle
    * **`'algebraic'`** calculates a solution that minimizes the regression error, where points are defined by the algebraic equation of a circle, a*x^T*x + b^T*x + c = 0
    * **`'bullock'`** calculates a solution that minimizes the geometric distance, defined by sum_i(g(u_i)^2), where g(u_i) = ||u_i - x||^2 - r^2
* solve (string): technique used to find the optimum defined by the algorithm
    * **`'exact'`** computes the exact solution using linear regression
    * **`'approximate'`** computes an approximate solution using gradient descent

#### Outputs:
* **`x` (dx1 np.matrix)** coordinates of the center of the circle
* **`r` (float)** radius of the circle

#### Failure:
* Raises a `ValueError` if the algorithm or the solve parameters are not one of the ones listed above
* Raises a `NotImplementedError` is solve is `'approximate'` -- gradient descent solutions haven't been implemented yet
* Running `singleCircle` on an empty matrix of points or fewer points than required to define a circle in the given dimensionality results in undefined behavior

### `xs, rs = fit.manyCircles(points, numCircles, algorithm='bullock', solve='exact')`
Uses a stochastic algorithm to fit multiple circles to a set of points by using some underlying algorithm and solving strategy to fit single circles to subsets of those points. Radii and centers of the circles are allowed to vary.

#### Parameters:
* **`points` (dxn np.matrix)** a matrix of column vectors containing the coordinates of each point to fit
* **`numCircles` (int)** number of circles to fit to the points
* **`algorithm` (string)** algorithm to use to fit each circle (see documentation for `singleCircle`)
* **`solve` (string)** solving strategy to use to fit each circle (see documentation for `singleCircle`)

#### Outputs:
* **`xs` (dxk np.matrix)** Column vectors containing the coordinates of the centers of each fitted circle
* **`rs` (1xk np.matrix)** Radius of each fitted circle

#### Failure:
* Check the failure conditions for `singleCircle`
