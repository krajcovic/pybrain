#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = 'krajcovic'

import numpy as np
import matplotlib.pyplot as plt

def line():
    plt.plot([1,2,3,4], [2, 4, 6, 8], linewidth=3.0)
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
    liner, lineb, lineg = plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g-')
    # print(liner, lineb)
    #lineg.set_antialiased(False)
    plt.show()

def multiplyfigures():

    def f(t):
        return np.exp(-t) * np.cos(2*np.pi*t)

    zoom = 1
    t1 = np.arange(0.0, 5.0 * zoom, 0.1 * zoom)
    t2 = np.arange(0.0, 5.0 * zoom, 0.02 * zoom)

    plt.figure(1)
    plt.subplot(211)
    plt.plot(t1, f(t1), 'bo', t2, f(t2), 'k')

    plt.subplot(212)
    plt.plot(t2, np.cos(2*np.pi*t2), 'r--')

    plt.show()

def histogram():
    mu, sigma = 100, 15
    x = mu + sigma * np.random.randn(10000)

    # the histogram of the data
    n, bins, patches = plt.hist(x, 50, normed=1, facecolor='g', alpha=0.75)


    plt.xlabel('Smarts')
    plt.ylabel('Probability')
    plt.title('Histogram of IQ')
    plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
    plt.axis([40, 160, 0, 0.03])
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    #line()
    # points()
    # multiplypoints()
    # multiplyfigures()
    histogram()