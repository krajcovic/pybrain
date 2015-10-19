#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = 'krajcovic'

from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import TanhLayer
from pybrain.structure import SoftmaxLayer

net1 = buildNetwork(2, 3, 1)
itteration = net1.activate([2,1])
print(itteration)

net2 = buildNetwork(2, 3, 1, hiddenclass=TanhLayer)
itteration = net2.activate([2,1])
print(itteration)

net3 = buildNetwork(2, 3, 1, hiddenclass=TanhLayer, outclass=SoftmaxLayer)
itteration = net3.activate([2,1])
print(itteration)

net4 = buildNetwork(2, 3, 1, bias=True)
itteration = net4.activate([2,1])
print(itteration)

net = buildNetwork(2, 3, 1, bias=True, hiddenclass=TanhLayer, fast=True)