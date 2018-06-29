import csv
from TabuSearch import TabuSearch
from random import shuffle
import time


def evaluate(solution):
    updatedDistances = [[0 for x in range(len(distanceCSV))] for y in range(len(distanceCSV))]

    for x in range(len(distanceCSV)):
        for y in range(len(distanceCSV)):
            updatedDistances[x][y] = distanceCSV[solution[x] - 1][solution[y] - 1]

    sum = 0

    for x in range(len(flowCSV)):
        for y in range(x + 1, len(flowCSV)):
            sum += int(flowCSV[x][y]) * int(updatedDistances[x][y])

    return sum

with open("Flow.csv") as flow:
    with open("Distance.csv") as distance:
        flowCSV = list(csv.reader(flow, delimiter=','))
        distanceCSV = list(csv.reader(distance, delimiter=','))

        initialSolution = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

        tabu = TabuSearch(tenure = 5,
                          candidatesNumber = 10,
                          maxIterations = 1000,
                          max_repetition = 10,
                          cost = evaluate,
                          initialSolution = initialSolution)

        tabu.search()
