import random

import numpy as np

def bisection_method(func, l, r, eps):
    def f(x):
        return eval(func)

    counter = 0

    # 1.
    xm = (l + r) / 2
    L = r - l
    fxm = f(xm)
    counter += 1

    i = 0
    while (1):
        i += 1

        # 2.
        x1 = l + L / 4
        x2 = r - L / 4
        fx1 = f(x1)
        counter += 1
        if fx1 >= fxm:
            fx2 = f(x2)
            counter += 1

        # 3.
        if fx1 < fxm:
            r = xm
            xm = x1
            fxm = fx1
            L = r - l
            if L < eps:
                break
        # 4.
        elif fx2 < fxm:
            l = xm
            xm = x2
            fxm = fx2
            L = r - l
            if L < eps:
                break
        # 5.
        else:
            l = x1
            r = x2
            L = r - l
            if L < eps:
                break
    print("***********************")
    print("Bisection method")
    print("Number of iterations: %d" % i)
    print("f(xm) =", fxm)
    print("xm =", xm)
    print("Number of functions calculated: %d" % counter)
    print("***********************")

def main():
    function = "((x ** 2 - 3) ** 2) / 9 - 1"
    # function = "(100 - x)**2"
    bisection_method(function, 0, 10, 0.0001)

if __name__ == '__main__':
    main()

