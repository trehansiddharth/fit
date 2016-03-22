import fit
import numpy as np

points1 = np.matrix([[1, 2, 5, 7, 9, 3], [7, 6, 8, 7, 5, 7]])
points2 = np.matrix([[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0], [0.0, 0.25, 1.0, 2.25, 4.0, 6.25, 9.0]])

offset = np.matrix([[15.0], [15.0]])

points3 = np.hstack((points1, points2 + offset))

print(fit.manyCircles(points3, 2, algorithm='bullock'))
