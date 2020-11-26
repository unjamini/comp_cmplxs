from scipy.optimize import linprog


c = [0, 0, 0, 1, 1, 1]
bounds = [(None, None), (None, None), (None, None), (0, None), (0, None), (0, None)]
A = [[-1, -10, 1, -1, 0, 0],
     [-1.2, -10, 1, 0, -1, 0],
     [-110, 0, 0, 0, 0, -1],
     [1, 10, -1, -1, 0, 0],
     [1.2, 10, -1, 0, -1, 0],
     [110, 0, 0, 0, 0, -1]]
r = [-2.5, -11.5, 0.5, 2.5, 11.5, -0.5]

res = linprog(c, A_ub=A, b_ub=r, bounds=bounds)
