import numpy as np

def onehot(i, k):
    if i < k and i > 0:
        v = np.matrix(np.zeros([k, 1]))
        v[i, 0] = 1
        return v
    else:
        return None

def gradient_descent(gradient_function, initial):
    theta = initial
    return theta

def linear_regression(A, b):
    theta, residuals, rank, s = np.linalg.lstsq(A, b)
    return theta

def norms(points):
    (numDimensions, numPoints) = np.shape(points)
    return np.matrix([np.linalg.norm(points[:,i]) ** 2 for i in range(0, numPoints)])

def centerOfMass(points):
    (numDimensions, numPoints) = np.shape(points)
    return points * (1 / float(numPoints)) * np.matrix(np.ones((numPoints, 1)))

def spread(points):
    (numDimensions, numPoints) = np.shape(points)
    transformedPoints = points - centerOfMass(points)
    ns = norms(transformedPoints)
    return np.sum(ns) / float(numPoints)

def listOf(x):
    return x.flatten().tolist()[0]

def normalize(xs):
    s = sum(xs)
    return map(lambda x: float(x) / float(s), xs)
