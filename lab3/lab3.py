import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from tolsolvty import tolsolvty


def draw_tol_plot():
    #  Функция для изображения графика распознающего функционала
    #  использует функцию calcfg из tolsolvty для подсчета значения
    x = np.arange(-1, 1, 0.01)
    y = np.arange(-1, 1, 0.01)
    z = np.zeros((x.shape[0], x.shape[0]))
    for i in range(x.shape[0]):
        for j in range(x.shape[0]):
            vec = np.array([[x[i]], [y[j]]])
            z[i, j] = tolsolvty.calcfg(vec)[0]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(30, 100)
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, shade=True)
    ax.plot([0], [-0.06], [z[100, 94]], markerfacecolor='c', markeredgecolor='c', marker='*', markersize=10)
    print(z[100, 94])
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('Tol(X1, X2)')
    # ax.grid(False)
    ax.xaxis.set_major_locator(plt.MaxNLocator(6))
    ax.yaxis.set_major_locator(plt.MaxNLocator(6))
    ax.zaxis.set_major_locator(plt.MaxNLocator(6))
    fig.colorbar(surf, shrink=0.5, aspect=5)
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_zticks([])
    plt.show()