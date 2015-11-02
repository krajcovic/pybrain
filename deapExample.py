#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'krajcovic'

import random
import time
from deap import base, creator, tools, algorithms
# from pylab import plt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def evalOneMax(individual):
    """
    Evaluation function.
    :param individual:
    :return: sum of individual.
    >>> evalOneMax([1])
    (1,)
    >>> evalOneMax([1, 2, 3])
    (6,)
    """
    return sum(individual),

def example_nevzpominam():
    # Creating appropriate type
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    # Initialization
    IND_SIZE = 100
    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=IND_SIZE)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evalOneMax)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=300)

    NGEN=40

    for gen in range(NGEN):
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.1)
        fits = toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        population = toolbox.select(offspring, k=len(population))
    top10 = tools.selBest(population, k=10)

def createType():
    """
    # Creating appropriate type

    :return:
    """
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

def initialization():
    """
    # Initialization

    :return:
    """
    IND_SIZE = 100
    toolbox = base.Toolbox()
    # toolbox.register("attr_bool", random.randint, 0, 1)
    # toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=IND_SIZE)
    toolbox.register("attribute", random.random)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=IND_SIZE)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evalOneMax)

    return toolbox



def easyComplete(toolbox):
    """
    Easy complete generational algorighm.

    :return:
    """
    pop = toolbox.population(n=20)
    CXPB, MUTPB, NGEN = 0.5, 0.2, 40

    # Evaluate the entire population
    fitness = map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitness):
        ind.fitness.values = fit

    for g in range(NGEN):
        #Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # print("Select len: ", len(pop))

        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))
        # print("Offspring len: ", len(offspring))
        # print(offspring)

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # The population is entirely replaced by the offspring
        pop[:] = offspring
        # print("Population :", len(pop[:]))
        # print(offspring)

    return pop[:]

def displayPopHistory(history):

    # fig = plt.figure()

    for pop in history:
        plt.clf()
        plt.scatter(list(range(len(pop))), pop, alpha=0.5)
        plt.show(block=True)
        print(pop)
        # time.sleep(1)


def animatePopHistory(history):

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # x = np.arange(0, 2*np.pi, 0.01)        # x-array
    x = np.arange(0, len(history[0]))
    line, = ax.plot(x, history[0])
    # x = np.arange(0, 20)
    # line, = ax.plot(x, x)

    def init():
        line.set_ydata(np.ma.array(x, mask=True))
        return line,

    def animate(i):
        # line.set_ydata(np.sin(x + i/10.0))  # update the data
        line.set_ydata(history[i])  # update the data
        return line,

    ani = animation.FuncAnimation(fig, animate, np.arange(1, 20), init_func=init, interval=25, blit=True)
    plt.show()




if __name__ == '__main__':
    import doctest
    doctest.testmod()

    createType()

    popHistory = easyComplete(initialization())
    displayPopHistory(popHistory)\
    # animatePopHistory(popHistory)
