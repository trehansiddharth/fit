import numpy as np
import math
import util

# x, r = singleCircle(points, algorithm='bullock', solve='exact'):
#   Fits a single circle to a set of points. Radius and center of the circle are
#   allowed to vary.
#   Parameters:
#       * points (dxn np.matrix): a matrix of column vectors containing the
#         coordinates of each point to fit
#       * algorithm (string): algorithm used to compute the best fit circle
#           - 'algebraic': calculates a solution that minimizes the regression
#             error, where points are defined by the algebraic equation of a
#             circle, a*x^T*x + b^T*x + c = 0
#           - 'bullock': calculates a solution that minimizes the geometric
#             distance, defined by sum_i(g(u_i)^2), where
#             g(u_i) = ||u_i - x||^2 - r^2
#       * solve (string): technique used to find the optimum defined by the
#         algorithm
#           - 'exact': computes the exact solution using linear regression
#           - 'approximate': computes an approximate solution using gradient
#             descent
#   Outputs:
#       * x (dx1): coordinates of the center of
#         the circle
#       * r (float): radius of the circle
#   Failure:
#       * Raises a ValueError if the algorithm or the solve parameters are not
#         one of the ones listed above
def singleCircle(points, algorithm='bullock', solve='exact'):
    if not (solve == 'exact' or solve == 'approximate'): raise ValueError
    if not (algorithm == 'algebraic' or algorithm == 'bullock'): raise ValueError
    # Record input dimensions
    numDimensions, numPoints = points.shape
    if algorithm == 'algebraic':
        # Compute the 2-norm of each point in the input
        norms = util.norms(points)

        # Define the matrices A and b we're going to use for optimization
        A = np.hstack((norms.T, points.T))
        b = np.ones((numPoints, 1))

        if solve == 'exact':
            # Run linear regression
            theta = util.linear_regression(A, b)
        else:
            raise NotImplementedError

        # Determine the parameters of the algebraic equation of the circle
        a = theta[0].item()
        b = theta[1:]
        c = -1

        # Determine the center and radius
        x = np.matrix(-b / (2.0 * a))
        r = math.sqrt(np.linalg.norm(x) ** 2 + 1 / a)
    else:
        # Transform the coordinates so that they are with respect to the center of mass
        (numDimensions, numPoints) = np.shape(points)
        center = util.centerOfMass(points)
        transformedPoints = points - center

        # Compute the norm of every point in the points matrix
        norms = util.norms(points)

        # Compute the matrices A and b to use in linear regression
        A = transformedPoints * transformedPoints.T
        b = 0.5 * transformedPoints * norms.T

        if solve == 'exact':
            # Run linear regression
            theta = util.linear_regression(A, b)
        else:
            raise NotImplementedError

        # Convert back to unshifted coordinate system and compute radius
        x = theta + center
        r = math.sqrt(np.linalg.norm(theta) ** 2 + np.sum(norms) / float(numPoints))

    return x, r

# xs, rs = manyCircles(points, numCircles, algorithm='bullock', solve='exact'):
#   Uses a stochastic algorithm to fit multiple circles to a set of points by
#   using some underlying algorithm and solving strategy to fit single circles
#   to subsets of those points. Radii and centers of the circles are allowed to
#   vary.
#   Parameters:
#       * points (dxn np.matrix): a matrix of column vectors containing the
#         coordinates of each point to fit
#       * numCircles (int): number of circles to fit to the points
#       * algorithm (string): algorithm to use to fit each circle (see
#         documentation for singleCircle)
#       * solve (string): solving strategy to use to fit each circle (see
#         documentation for singleCircle)
#   Output:
#       * xs (dxk np.matrix): Column vectors containing the coordinates of the
#         centers of each fitted circle
#       * rs (1xk np.matrix): Radius of each fitted circle
#   Failure:
#       * Check the failure conditions for singleCircles
def manyCircles(points, numCircles, algorithm='bullock', solve='exact'):
    # Compute parameters that help choose good initial conditions
    (numDimensions, numPoints) = np.shape(points)
    m = util.centerOfMass(points)
    s = util.spread(points)

    # Choose initial centers
    cov = s * np.identity(numDimensions)
    xs = np.matrix(np.random.multivariate_normal(util.listOf(m), cov, numCircles).T)

    # Choose initial radii
    rs = np.matrix(np.random.exponential(0.5 * s, numCircles))

    # Initialize lambda parameter
    l = 2.0 * s

    # Initialize matrices of errors to zero (to start off with uniform point assignment probabilities)
    E = np.zeros((numCircles, numPoints))

    # Main loop
    while True:
        # Stochastically choose assignments of points to circles
        L = np.zeros(numPoints)
        for i in range(0, numPoints):
            ps = util.normalize(map(lambda e: math.exp(- e / l), E[:,i]))
            assignment = np.random.choice(numCircles, p=ps)
            L[i] = assignment

        # Rearrange the points matrix to group points in the same assignment together
        sortOrder = np.argsort(L)
        L = np.hstack([L[i] for i in sortOrder])
        points = np.hstack([points[:,i] for i in sortOrder])

        # Find the boundaries between assignment labels in the sorted list
        groupBoundaries = []
        for i in range(0, numPoints + 1):
            if i == 0:
                extension = L[0]
            elif i == numPoints:
                extension = numCircles - 1 - L[numPoints - 1]
            else:
                previousAssignment = L[i - 1]
                thisAssignment = L[i]
                extension = thisAssignment - previousAssignment
            groupBoundaries.extend([i] * extension)

        # Split the points matrix into smaller matrices based on these group boundaries
        pointMatrices = np.split(points, groupBoundaries, axis=1)

        # Compute the best fitting circle for each point matrix
        for i in range(0, numCircles):
            if pointMatrices[i].size != 0:
                x, r = singleCircle(pointMatrices[i], algorithm, solve)
                xs[:,i] = x
                rs[:,i] = r

        # Recompute errors
        for i in range(0, numCircles):
            for j in range(0, numPoints):
                x = xs[:,i]
                r = rs[:,i]
                p = points[:,j]

                # E[i, j] is how far the jth point is from the border of the ith circle
                E[i, j] = abs(np.linalg.norm(p - x) - r)

        # Reduce lambda parameters
        l = l * 0.9

        # Exit the loop if lambda becomes too small
        if l / s < 0.001: break

    return xs, rs
