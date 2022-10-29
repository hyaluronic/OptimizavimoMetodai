import itertools
import math
from math import sin
from operator import itemgetter
import random

import matplotlib.pyplot as plt
import numpy as np
from sympy import Symbol

fig, ax = plt.subplots()


def f(x1, x2):
    return gradientFunction(str(F), x1, x2)

def plot2d(points, method):
    plot2d(points, [], method)

def plot2d(points, pointsForNumbers, method):
    delta = 0.001
    if method == 'gd' or method == 'sd':
        x = np.arange(0.15, 0.35, delta)
        y = np.arange(0.30, 0.5, delta)
        X, Y = np.meshgrid(x, y)
        Z = -0.125 * X * Y * (1 - X - Y)
        CS = ax.contour(X, Y, Z, 15, linewidths=0.3)
        ax.clabel(CS, inline=True, fontsize=9)
        iterations = [2, 3, 4, 5, 6]
        for i in range(0, len(points), 2):
            if int(i / 2) in iterations:
                if i == (len(points) - 2):
                    ax.plot(points[i], points[i + 1], marker='x')
                else:
                    ax.plot(points[i], points[i + 1], marker='.')
                plt.annotate(int(i / 2) + 1, (points[i], points[i + 1] + 0.003))
    elif method == 'simplex' and len(points) % 3 == 0:
        for i in range(0, min(5*3, len(points)), 3):
            simp = [points[i], points[i + 1], points[i + 2]]
            r = random.random()
            b = random.random()
            g = random.random()
            a = 1
            color = (r, g, b, a)
            for a, b in itertools.product(simp, simp):
                x = np.linspace(a[0], b[0], 100)
                y = np.linspace(a[1], b[1], 100)

                ax.plot(x, y, color=color)
        for i in range(0, min(7, len(pointsForNumbers))):
            plt.annotate(i + 1, (pointsForNumbers[i][0], pointsForNumbers[i][1] + 0.009), fontsize=15)
    plt.draw()
    plt.show()


def getGrad(func, args):
    grad = []
    for x in args:
        grad.append(func.diff(x))
    return grad


def gradientFunction(func, x1, x2):
    return eval(func)


def gradient_descent(func, args, X0, gama, epsilon):
    i = 1
    Xi = X0

    grad = getGrad(func, args)
    points = []
    counter = 0

    max_iterations = 100
    while i < max_iterations:
        gradMod = 0
        Xtemp = list(Xi)
        for j in range(0, len(Xi)):
            gradFunc = gradientFunction(str(grad[j]), Xtemp[0], Xtemp[1])
            Xi[j] = Xi[j] - gama * gradFunc
            counter += 1
            gradMod += gradFunc
        gradMod = abs(gradMod) / len(Xi)
        points.append(Xi[0])
        points.append(Xi[1])
        print("i: ", i, "Xi[0]:", Xi[0], ". Xi[1]:", Xi[1], ". f(Xi) =", f(Xi[0], Xi[1]))
        if gradMod < epsilon:
            print("i: ", i, "[", Xi[0], ",", Xi[1], "]")
            print("Counter: ", counter)
            print("f(X) = ", f(Xi[0], Xi[1]))
            break
        i += 1
    plot2d(points, 'gd')

def golden_section_method(left, right, xi, grad, func, epsilon):
    f = lambda alpha: fPoint(xi - alpha * grad)

    def fPoint(point):
        return gradientFunction(str(func), point[0], point[1])

    tau = (-1 + math.sqrt(5)) / 2
    length = right - left
    x1 = right - tau * length
    x2 = left + tau * length
    steps = 1

    fx1 = f(x1)
    fx2 = f(x2)
    counter = 2
    while length >= epsilon:
        steps = steps + 1
        if fx2 < fx1:
            left = x1
            length = right - left
            x1 = x2
            fx1 = fx2
            x2 = left + tau * length
            fx2 = f(x2)
            counter += 1
        else:
            right = x2
            length = right - left
            x2 = x1
            fx2 = fx1
            x1 = right - tau * length
            fx1 = f(x1)
            counter += 1

        min_reiksme = min([fx1, fx2])
        sprendinys = x1
        if min_reiksme == fx2:
            sprendinys = x2
    return sprendinys, counter

def steepest_descent(func, args, Xi, epsilon):
    Xi = np.array(Xi)

    grad = getGrad(func, args)
    points = []
    # points = [Xi[0], Xi[1]]

    counter = 0
    i = 1
    max_iterations = 500
    while i < max_iterations:
        gradValue = np.array(gradientFunction(str(grad), Xi[0], Xi[1]))
        counter += 2
        gradMod = math.sqrt(gradValue[0] * gradValue[0] + gradValue[1] * gradValue[1])
        if gradMod < epsilon:
            print("i: ", i - 1, "[", Xi[0], ",", Xi[1], "]")
            print("Counter: ", counter)
            print("f(X) = ", f(Xi[0], Xi[1]))
            break

        gama_min, n_count = golden_section_method(0, 17, Xi, gradValue, func, epsilon)
        counter += n_count

        Xi = Xi - gama_min * gradValue
        points.append(Xi[0])
        points.append(Xi[1])
        print("i: ", i, "gamma", gama_min, "Xi[0]:", Xi[0], ". Xi[1]:", Xi[1], ". f(Xi) =", f(Xi[0], Xi[1]))
        i += 1
    plot2d(points, 'sd')

def getModVector(x):
    return math.sqrt(x[0] * x[0] + x[1] * x[1])
def getPoint(arg, value):
    return {"arg": arg, "value": value}

def generate_simplex_method_points(x0, alpha):
    n = 2
    delta1 = (math.sqrt(n + 1) + n - 1) / (n * math.sqrt(2)) * alpha
    delta2 = (math.sqrt(n + 1) - 1) / (n * math.sqrt(2)) * alpha

    x1 = [x0[0] + delta2, x0[0] + delta1]
    x2 = [x0[0] + delta1, x0[0] + delta2]
    return x1, x2

def simplex_method(args, X0, epsilon=0.001, alpha=0.5, gama=3, beta=0.2, niu=-0.7):
    simplex = [getPoint(X0, f(X0[0], X0[1]))]
    max_iterations = 100
    counter = 1
    points = []
    pointsForNumbers = [X0]

    for i in range(0, len(args)):
        argList = list(X0)
        argList[i] -= alpha
        simplex.append(getPoint(argList, f(argList[0], argList[1])))
        counter += 1

    # Geresnis:
    # X1, X2 = generate_simplex_method_points(X0, alpha)
    # simplex.append(getPoint(X1, f(X1[0], X1[1])))
    # counter += 1
    # simplex.append(getPoint(X2, f(X2[0], X2[1])))
    # counter += 1

    pointsForNumbers.append(simplex[-2]['arg'])

    for i in range(0, max_iterations):
        pointsForNumbers.append(simplex[-1]['arg'])
        # 1. Sort
        simplex.sort(key=itemgetter('value'))

        print("i: ", i + 1)
        print("     [0]: ("+str(simplex[0]['arg'][0])+", "+str(simplex[0]['arg'][1])+")", ". f:", simplex[0]['value'])
        # print("     [1]:", simplex[1]['arg'], ". f:", simplex[1]['value'])
        # print("     [2]:", simplex[2]['arg'], ". f:", simplex[2]['value'])

        # if i < 6:
        points.extend([tuple(sim['arg'] + [sim['value']]) for sim in simplex])

        # 6. Check convergence
        if getModVector(np.array((simplex[0]['arg']) - np.array(simplex[-1]['arg']))) < epsilon:
            break

        centroid = [0] * len(args)
        for j in range(0, len(args)):
            for k in range(0, len(simplex) - 1):
                centroid[j] += simplex[k]['arg'][j]
            centroid[j] /= (len(simplex) - 1)

        # 2. Reflect
        reflection = [0] * len(args)
        for j in range(0, len(args)):
            reflection[j] = centroid[j] + alpha * (centroid[j] - simplex[-1]['arg'][j])
        reflection_value = f(reflection[0], reflection[1])
        counter += 1

        # 3. Evaluate or Extend
        if simplex[0]['value'] <= reflection_value < simplex[-2]['value']:
            simplex[-1] = getPoint(reflection, reflection_value)
            continue
        elif reflection_value < simplex[0]['value']:
            extend = [0] * len(args)
            for j in range(0, len(args)):
                extend[j] = centroid[j] + gama * (reflection[j] - centroid[j])
            extended_value = f(extend[0], extend[1])
            counter += 1
            if extended_value < simplex[0]['value']:
                simplex[-1] = getPoint(extend, extended_value)
            else:
                simplex[-1] = getPoint(reflection, reflection_value)
            continue

        # 4. Contract
        contraction = [0] * len(args)
        for j in range(0, len(args)):
            contraction[j] = centroid[j] + niu * (simplex[-1]['arg'][j] - centroid[j])
        contraction_value = f(contraction[0], contraction[1])
        counter += 1
        if contraction_value < simplex[-1]['value']:
            simplex[-1] = getPoint(contraction, contraction_value)
            continue

        # 5. Reduce
        for j in range(1, len(simplex)):
            reduce = [0] * len(args)
            for k in range(0, len(args)):
                reduce[k] = simplex[0]['arg'][k] + beta * (simplex[j]['arg'][k] - simplex[0]['arg'][k])
            reduce_value = f(reduce[0], reduce[1])
            counter += 1
            simplex[j] = getPoint(reduce, reduce_value)
        pointsForNumbers.append(simplex[-2]['arg'])

    plot2d(points, pointsForNumbers, 'simplex')
    print("i: ", i + 1, simplex[0]['arg'])
    print("f(X) = ", f(simplex[0]['arg'][0], simplex[0]['arg'][1]))
    print("Counter: ", counter)


x1 = Symbol('x1')
x2 = Symbol('x2')
F = -0.125 * x1 * x2 * (1 - x1 - x2)
def main():

    # plot2d([], 'a')
    #
    # grad = getGrad(F, [x1, x2])
    # print("Tikslo ir gradiento funkciju reiksmes (0, 0)", f(0, 0), gradientFunction(str(grad), 0, 0))
    # print("Tikslo ir gradiento funkciju reiksmes (1, 1)", f(1, 1), gradientFunction(str(grad), 1, 1))
    # print("Tikslo ir gradiento funkciju reiksmes (0.3, 0.9)", f(0.3, 0.9), gradientFunction(str(grad), 0.3, 0.9))

    # gradient_descent(F, [x1, x2], [0, 0], 0.1, 0.001)
    # gradient_descent(F, [x1, x2], [1, 1], 3.6, 0.001)
    # gradient_descent(F, [x1, x2], [0.3, 0.9], 3.4, 0.0001)

    # steepest_descent(F, [x1, x2], [0, 0], 0.001)
    # steepest_descent(F, [x1, x2], [1, 1], 0.001)
    # steepest_descent(F, [x1, x2], [0.3, 0.9], 0.001)

    # simplex_method([x1, x2], [0, 0])
    # simplex_method([x1, x2], [1, 1])
    simplex_method([x1, x2], [0.3, 0.9])


if __name__ == "__main__":
    main()