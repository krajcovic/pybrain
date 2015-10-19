#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = 'krajcovic'

from pybrain.datasets import SupervisedDataSet

ds = SupervisedDataSet(2,1)


def init():
    ds.addSample((0,0), (0,))
    ds.addSample((0,1), (1,))
    ds.addSample((1,0), (1,))
    ds.addSample((1,1), (0,))

    # for inp, target in ds:
    #     print(inp, target)


def clear():
    ds.clear()


def show():
    print("Input\n", ds['input'])
    print("Target\n", ds['target'])


if __name__ == "__main__":
    init()
    show()
else:
    clear()
    init()
