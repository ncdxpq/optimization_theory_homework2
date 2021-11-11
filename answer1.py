import numpy as np
import matplotlib.pyplot as plt


def get_matrix(a, b, c):
    """
    逼近平面y=ax+by+c附近的188个随机点,转化为矩阵形式Ax=b
    :param a:x的系数
    :param b:y的系数
    :param c:常数项c
    :return: 矩阵系数A，b
    """
    # 创建函数，用于随机生成不同属于一个平面的188个离散点
    # np.random.uniform从均匀分布中随机采样
    x = np.random.uniform(-10, 10, size=188)
    y = np.random.uniform(-10, 10, size=188)
    z = (a * x + b * y + c) + np.random.normal(-0.5, 0.5, size=188)  # 加上了随机噪音

    # 创建系数矩阵A和b
    A = np.ones((188, 3))  # 188*3矩阵，全是1
    b = np.zeros((188, 1))  # 188*1矩阵，全是1
    for i in range(0, 188):  # A的前两列全部赋值为点x和y，第三列为1。b由z的值确定
        A[i, 0] = x[i]
        A[i, 1] = y[i]
        b[i, 0] = z[i]
    return A, b, x, y, z


def matrix_compute(A, b):
    """X=(AT*A)-1*AT*b直接求解"""
    A_T = A.T  # 获得矩阵A的转置(3*188)
    AT_A = np.dot(A_T, A)  # 矩阵乘法(3*188)*(188*3)=(3*3)
    AT_A_reverse = np.linalg.inv(AT_A)  # 矩阵求逆(3*3)
    AT_A_reverse_A_T = np.dot(AT_A_reverse, A_T)  # (3*3)*(3*188)=(3*188)
    X = np.dot(AT_A_reverse_A_T, b)  # (3*188)*(188*1)=(3*1)
    print('最小二乘法拟合的最终平面为：z = %.2f * x + %.2f * y + %.2f' % (X[0, 0], X[1, 0], X[2, 0]))
    return X  # 获得系数a,b,c


if __name__ == '__main__':
    # 获得题干的函数z=2x+8y+10，将其系数填入矩阵以及获得该188个点
    A, b, x, y, z = get_matrix(2, 8, 10)
    X = matrix_compute(A, b)  # 最小二乘法运算，获得未知系数a,b,c
    # 展示图像
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.set_xlabel("x")  # 标签
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")
    ax1.scatter(x, y, z, c='b', marker='o')  # 画出188个点的散点图
    x_p = np.linspace(-10, 10, 188)
    y_p = np.linspace(-10, 10, 188)
    x_p, y_p = np.meshgrid(x_p, y_p)
    z_p = X[0, 0] * x_p + X[1, 0] * y_p + X[2, 0]  # 画出最小二乘法拟合的平面
    ax1.plot_wireframe(x_p, y_p, z_p, rstride=15, cstride=15, color='r')
    plt.show()
