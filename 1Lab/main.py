import random

import numpy as np
from matplotlib import pyplot as plt

bisection_method_points = []
golden_section_method_points = []
newtons_method_points = []

def bisection_method(func, l, r, eps):
    def f(x):
        return eval(func)

    counter = 0

    # 1.
    xm = (l + r) / 2
    L = r - l
    fxm = f(xm)
    counter += 1

    for i in range(1, 100):
        # 2.
        x1 = l + L / 4
        x2 = r - L / 4
        fx1 = f(x1)
        counter += 1
        if fx1 >= fxm:
            fx2 = f(x2)
            counter += 1

        # 3. Atmetamas (xm , r]
        if fx1 < fxm:
            r = xm
            xm = x1
            fxm = fx1
            L = r - l
        # 4. Atmetamas [l, xm)
        elif fx2 < fxm:
            l = xm
            xm = x2
            fxm = fx2
            L = r - l
        # 5. Atmetamas [l, x1) ir (x2, r]
        else:
            l = x1
            r = x2
            L = r - l

        bisection_method_points.append([l, r])

        # 6.
        if L < eps:
            break

    print("***********************")
    print("Bisection method")
    print("Number of iterations: %d" % i)
    print("f(Xmin) =", fxm)
    print("Xmin =", xm)
    print("Number of functions calculated: %d" % counter)
    print("***********************")

def golden_section_method(func, l, r, eps, tau):
    counter = 0
    def f(x):
        return eval(func)

    # 1.
    L = r - l
    x1 = r - tau * L
    x2 = l + tau * L
    fx1 = f(x1)
    fx2 = f(x2)
    counter += 2

    for i in range(1, 100):
        # 2. Atmetamas [l, x1)
        if fx2 < fx1:
            l = x1
            L = r - l
            x1 = x2
            fx1 = fx2
            x2 = l + tau * L
            fx2 = f(x2)
            counter += 1
        # 3. Atmetamas (x2, r]
        else:
            r = x2
            L = r - l
            x2 = x1
            fx2 = fx1
            x1 = r - tau * L
            fx1 = f(x1)
            counter += 1

        golden_section_method_points.append([l, r])

        # 4.
        if L < eps:
            break

    print("***********************")
    print("Golden section method")
    print("Number of iterations: %d" % i)
    print("f(Xmin) =", fx1 if fx1 < fx2 else fx2)
    print("Xmin =", x1 if fx1 < fx2 else x2)
    print("Number of functions calculated: %d" % counter)
    print("***********************")

def newtons_method(func, x0, eps):
    counter = 0

    for i in range(1, 100):
        xn = x0 - first_deriv(x0, func) / second_deriv(x0, func)
        counter += 2
        newtons_method_points.append(xn)
        if abs(xn - x0) < eps:
            break
        x0 = xn

    print("***********************")
    print("Newton's method")
    print("Number of iterations: %d" % i)
    print("f(Xmin): %f" % f(xn, func))
    print("Xmin: %f" % xn)
    print("Number of functions calculated: %d" % counter)
    print("***********************")

def f(x, func):
    return eval(func)

def first_deriv(x, func):
    h = 1e-5
    return (f(x + h, func) - f(x, func)) / h

def second_deriv(x, func):
    h = 1e-5
    return (first_deriv(x + h, func) - first_deriv(x, func)) / h

def drawGraph(func, l, r, char):
    x = np.linspace(l, r, 100)
    y = eval(func)

    print("bisection: ", bisection_method_points)
    print("golden: ", golden_section_method_points)
    print("newtons: ", newtons_method_points)

    fig1 = plt.figure()
    ax = fig1.add_subplot(1, 1, 1)
    ax.set(ylim=(-2, 5))
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    if char == 'b':
        drawPoints(bisection_method_points)
    elif char == 'g':
        drawPoints(golden_section_method_points)
    elif char == 'n':
        drawPoints(newtons_method_points)

    plt.plot(x, y, 'r')
    plt.show()

def drawPoints(point_array):
    show = [1, 2, 3, 4, len(point_array)]
    for i in range(0, len(point_array)):
        r = random.random()
        b = random.random()
        g = random.random()
        a = 1
        color = (r, g, b, a)
        if i+1 in show:
            if isinstance(point_array[i], list):
                plt.scatter(point_array[i][0], 0, color=color)
                plt.annotate(i + 1, (point_array[i][0], 0))
                plt.scatter(point_array[i][1], 0, color=color)
                plt.annotate(i + 1, (point_array[i][1], 0))
            else:
                plt.scatter(newtons_method_points[i], 0, color=color)
                plt.annotate(i + 1, (newtons_method_points[i], 0))

def main():
    function = "((x ** 2 - 3) ** 2) / 9 - 1"
    l = 0
    r = 10
    # function = "(100 - x) ** 2"
    # l = 60
    # r = 150
    eps = 0.0001
    bisection_method(function, l, r, eps)
    tau = 0.61803
    golden_section_method(function, l, r, eps, tau)
    x0 = 5
    newtons_method(function, x0, eps)
    drawGraph(function, 0, 4, 'n')

if __name__ == '__main__':
    main()

