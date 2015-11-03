#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = 'krajcovic'

import numpy as np
import matplotlib.pyplot as plt

def line():
    plt.plot([1,2,3,4], [2, 4, 6, 8])
    plt.ylabel('some number')
    plt.show()

def points():
    plt.plot([1,2,3,4], [2, 4, 6, 8], 'bo')
    plt.ylabel('some number')
    plt.show()

def multiplypoints():
    # evenly sampled time at 200ms intervals
    t = np.arange(0., 5., 0.2)

    # red dashes, blue squares and green triangles
    plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
    plt.show()

if __name__ == "__main__":
    # line()
    # points()
    multiplypoints()