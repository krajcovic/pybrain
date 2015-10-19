#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = 'krajcovic'

from pybrain.supervised.trainers import BackpropTrainer
from buildingDataset import ds
from buildingNetwork import net
from pybrain.tools.validation import CrossValidator

trainer = BackpropTrainer(net, ds)

fullEpoch = trainer.train()
print(fullEpoch)

covergence = trainer.trainUntilConvergence()
print(covergence)

# cv = CrossValidator(trainer, ds)