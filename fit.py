import numpy as np
import math
import util

class Shape:
    def __init__(self):
        pass

    def single(self, points):
        return self.get(self._single(points))

    def many(self, points, numShapes):
        # Compute parameters that help choose good initial conditions
        (numDimensions, numPoints) = np.shape(points)

        # Choose initial theta parameters
        thetas = self.initial_theta_function(points, numShapes)

        # Initialize lambda parameter
        l = self.initial_lambda_function(points, numShapes)

        # Initialize matrices of errors to zero (to start off with uniform point assignment probabilities)
        E = np.zeros((numShapes, numPoints))

        # Main loop
        while True:
            # Stochastically choose assignments of points to shapes
            L = np.zeros(numPoints)
            for i in range(0, numPoints):
                # print(map(lambda e: math.exp(- e / l), E[:,i]))
                ps = util.normalize(map(lambda e: math.exp(- e / l), E[:,i]))
                assignment = np.random.choice(numShapes, p=ps)
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
                    extension = numShapes - 1 - L[numPoints - 1]
                else:
                    previousAssignment = L[i - 1]
                    thisAssignment = L[i]
                    extension = thisAssignment - previousAssignment
                groupBoundaries.extend([i] * extension)

            # Split the points matrix into smaller matrices based on these group boundaries
            pointMatrices = np.split(points, groupBoundaries, axis=1)

            # Compute the best fitting shape for each point matrix
            for i in range(0, numShapes):
                if pointMatrices[i].size != 0:
                    theta = self._single(pointMatrices[i])
                    thetas[:,i] = theta

            # Recompute errors
            for i in range(0, numShapes):
                for j in range(0, numPoints):
                    theta = thetas[:,i]
                    p = points[:,j]

                    # E[i, j] is how far the jth point is from the border of the ith circle
                    E[i, j] = self.error_function(p, theta)

            # Reduce lambda parameters
            l = l * 0.9

            # Exit the loop if lambda becomes too small
            if l < 0.01: break

        return self.getMany(thetas)

class Circle(Shape):
    def __init__(self, algorithm='bulock', solve='exact'):
        if not (solve == 'exact' or solve == 'approximate'): raise ValueError
        if not (algorithm == 'algebraic' or algorithm == 'bullock'): raise ValueError
        self.algorithm = algorithm
        self.solve = solve
        Shape.__init__(self)

    def get(self, theta):
        return theta[0:-1], theta[-1].item()

    def getMany(self, thetas):
        return thetas[0:-1,:], thetas[-1,:]

    def error_function(self, p, theta):
        x, r = self.get(theta)
        # The distance between the point and the edge of the circle
        return abs(np.linalg.norm(p - x) - r)

    def initial_theta_function(self, points, numCircles):
        # Compute parameters that help choose good initial conditions
        (numDimensions, numPoints) = np.shape(points)
        m = util.centerOfMass(points)
        s = util.spread(points)

        # Choose initial centers
        cov = s * np.identity(numDimensions)
        xs = np.matrix(np.random.multivariate_normal(util.listOf(m), cov, numCircles).T)

        # Choose initial radii
        rs = np.matrix(np.random.exponential(0.5 * s, numCircles))

        return np.vstack((xs, rs))

    def initial_lambda_function(self, points, numCircles):
        # Compute parameters that help choose good initial conditions
        (numDimensions, numPoints) = np.shape(points)
        m = util.centerOfMass(points)
        s = util.spread(points)

        return 2.0 * s

    def _single(self, points):
        # Record input dimensions
        numDimensions, numPoints = points.shape
        if self.algorithm == 'algebraic':
            # Compute the 2-norm of each point in the input
            norms = util.norms(points)

            # Define the matrices A and b we're going to use for optimization
            A = np.hstack((norms.T, points.T))
            b = np.ones((numPoints, 1))

            if self.solve == 'exact':
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

            if self.solve == 'exact':
                # Run linear regression
                theta = util.linear_regression(A, b)
            else:
                raise NotImplementedError

            # Convert back to unshifted coordinate system and compute radius
            x = theta + center
            r = math.sqrt(np.linalg.norm(theta) ** 2 + np.sum(norms) / float(numPoints))

        return np.vstack((x, np.matrix([[r]])))
