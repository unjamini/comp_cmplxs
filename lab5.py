import numpy as np

A = np.array([
    [[2, 4], [-2, 1]],
    [[-2, 1], [2, 4]]
    ])

b_low = np.array([-1, -1])
b_high = np.array([1, 1])


def interval_mul(x_low, x_high, y_low, y_high):
    return max(x_low * y_low, x_high * y_high), min(x_low * y_high, x_high * y_low)


def apply_sti(vec_inf, vec_sup):
    return np.append(-vec_inf, vec_sup)


def reverse_sti(vec):
    mid_idx = vec.shape[0] // 2
    return -vec[:mid_idx], vec[mid_idx:]


def get_sti_matrix(matrix):
    pos = matrix.copy()
    neg = matrix.copy()
    pos[pos < 0] = 0
    neg[neg > 0] = 0
    neg = np.fabs(neg)
    return np.block([[pos, neg], [neg, pos]])


def sum_intervals(interval_list):
    result = [0, 0]
    for interval in interval_list:
        result[0] += interval[0]
        result[1] += interval[1]
    return result[0], result[1]


def get_func_value(cur_vec, C, d):
    inf, sup = reverse_sti(cur_vec)
    ml = [sum_intervals([interval_mul(C[i][j][0], C[i][j][1], inf[j], sup[j]) for j in range(len(inf))]) for i in range(len(C))]
    low_matr = get_low_matrix(ml)
    high_matr = get_high_matrix(ml)
    return apply_sti(low_matr, high_matr) - d


def get_sub_grad(prev_D, A, n, x):
    mid = prev_D.shape[0] // 2
    for i in range(n):
        for j in range(n):
            val_low, val_high = A[i, j]
            b_low = -x[j]
            b_high = x[j + n]
            if val_low * val_high > 0:
                k = 0 if val_low > 0 else 2
            else:
                k = 1 if val_low < val_high else 3

            if b_low * b_high > 0:
                m = 1 if b_low > 0 else 3
            else:
                m = 2 if b_low <= b_high else 4

            case = 4 * k + m
            if case == 1:
                prev_D[i, j] = val_low
                prev_D[i + mid, j + mid] = val_high
            elif case == 2:
                prev_D[i, j] = val_high
                prev_D[i + mid, j + mid] = val_high
            elif case == 3:
                prev_D[i, j] = val_high
                prev_D[i + mid, j + mid] = val_low
            elif case == 4:
                prev_D[i, j] = val_low
                prev_D[i + mid, j + mid] = val_low
            elif case == 5:
                prev_D[i, j + mid] = val_low
                prev_D[i + mid, j + mid] = val_high
            elif case == 6:
                if val_low * b_high < val_high * b_low:
                    prev_D[i, j + mid] = val_low
                else:
                    prev_D[i, j] = val_high
                if val_low * b_low > val_high * b_high:
                    prev_D[i + mid, j] = val_low
                else:
                    prev_D[i + mid, j + mid] = val_high
            elif case == 7:
                prev_D[i, j] = val_high
                prev_D[i + mid, j] = val_low
            elif case == 8:
                pass
            elif case == 9:
                prev_D[i, j + mid] = val_low
                prev_D[i + mid, j] = val_high
            elif case == 10:
                prev_D[i, j + mid] = val_low
                prev_D[i + mid, j] = val_low
            elif case == 11:
                prev_D[i, j + mid] = val_high
                prev_D[i + mid, j] = val_low
            elif case == 12:
                prev_D[i, j + mid] = val_high
                prev_D[i + mid, j] = val_high
            elif case == 13:
                prev_D[i, j] = val_low
                prev_D[i + mid, j] = val_high
            elif case == 14:
                pass
            elif case == 15:
                prev_D[i, j + mid] = val_high
                prev_D[i + mid, j + mid] = val_low
            elif case == 16:
                if val_low * b_low > val_high * b_high:
                    prev_D[i, j] = val_low
                else:
                    prev_D[i, j + mid] = -val_high
                if val_low * b_high < val_high * b_low:
                    prev_D[i + mid, j + mid] = val_low
                else:
                    prev_D[i + mid, j] = val_high
    return prev_D


def get_low_matrix(interval_matrix):
    func = lambda x: x[0]
    return np.array(list(func(interval_matrix)))


def get_high_matrix(interval_matrix):
    func = lambda x: x[1]
    return np.array(list(func(interval_matrix)))


def get_mid_matrix(interval_matrix):
    func = lambda x: (x[1] + x[0]) / 2
    return func(interval_matrix)


def sub_grad_2(A, b_inf, b_sup, acc=1e-5, param=1):
    n = A.shape[0]
    sti_b = apply_sti(b_inf, b_sup)
    x = np.zeros_like(sti_b)

    prev_x = x
    iter_count = 0
    while (not iter_count or np.linalg.norm(x - prev_x) > acc) and iter_count < 100:
        iter_count += 1
        prev_x = x
        subgrad_D = np.zeros((2 * n, 2 * n))
        subgrad_D = get_sub_grad(subgrad_D, A, n, prev_x)
        func_v = get_func_value(x, A, sti_b)
        dx = np.linalg.solve(subgrad_D, -func_v)
        x = prev_x + param * dx
    return reverse_sti(x), iter_count


# print(reverse_sti(apply_sti(b_inf, b_sup)))
# print(get_sti_matrix(A))

(x_inf, x_sup), iters = sub_grad_2(A, b_low, b_high)
print(f"X_inf: {x_inf}")
print(f"X_sup: {x_sup}")
print(f"Iterations number: {iters}")
