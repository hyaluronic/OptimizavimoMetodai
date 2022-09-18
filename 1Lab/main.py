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

        # 6.
        if L < eps:
            break

    print("***********************")
    print("Bisection method")
    print("Number of iterations: %d" % i)
    print("f(xm) =", fxm)
    print("xm =", xm)
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
    i = 0
    while (1):
        i += 1
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

        # 4.
        if L < eps:
            break

    print("***********************")
    print("Golden section method")
    print("Number of iterations: %d" % i)
    findMin(fx1, fx2, x1, x2)
    print("Number of functions calculated: %d" % counter)
    print("***********************")

def findMin(fx1, fx2, x1, x2):
    if fx1 < fx2:
        print("f(x1) =", fx1)
        print("x1 =", x1)
    else:
        print("f(x2) =", fx2)
        print("x2 =", x2)

def main():
    # function = "((x ** 2 - 3) ** 2) / 9 - 1"
    # l = 0
    # r = 10
    function = "(100 - x)**2"
    l = 60
    r = 150
    eps = 0.0001
    bisection_method(function, l, r, eps)
    tau = 0.61803
    golden_section_method(function, l, r, eps, tau)

if __name__ == '__main__':
    main()

