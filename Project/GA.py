from Tensorflow.tensorflow_mnist import TensorFlowMNIST

import sys;
import random;
import numpy as np;
import heapq;
import tensorflow as tf


#CONSTANTS
MUTATION_PROPABILITY = 0.02;
CROSSOVER_ALPHA = 0.8;
POPULATION_SIZE = 6;
MAX_ITERATIONS = 100;
DATA_PATH = "MNIST_data/mnist.npz"
TF_EP = 3

HID_LAY_LIMITS = [1, 5]
HID_NODE_LIMITS = [8, 256]
LR_LIMITS = [0.001, 0.05]

#GLOBALS
generation = 0;

def main():
    global generation;
    generation = 0;
    population = [];

    individual = createIndividual({
        "hid_lay": 1,
        "hid_node": 256,
        "lr": 0.01
    });
    population.append({
        "individual": individual,
        "value": evaluateIndividual(individual)
    });
    individual = createIndividual({
        "hid_lay": 2,
        "hid_node": 256,
        "lr": 0.05
    });
    population.append({
        "individual": individual,
        "value": evaluateIndividual(individual)
    });
    individual = createIndividual({
        "hid_lay": 1,
        "hid_node": 256,
        "lr": 0.04
    });
    population.append({
        "individual": individual,
        "value": evaluateIndividual(individual)
    });
    individual = createIndividual({
        "hid_lay": 2,
        "hid_node": 256,
        "lr": 0.01
    });
    population.append({
        "individual": individual,
        "value": evaluateIndividual(individual)
    });
    individual = createIndividual({
        "hid_lay": 1,
        "hid_node": 256,
        "lr": 0.03
    });
    population.append({
        "individual": individual,
        "value": evaluateIndividual(individual)
    });
    individual = createIndividual({
        "hid_lay": 3,
        "hid_node": 128,
        "lr": 0.05
    });
    population.append({
        "individual": individual,
        "value": evaluateIndividual(individual)
    });

    population.sort(reverse = True, key=evaluatePopulation)

    cycleOfLife(population);

# Contains all function calls for each cycle
def cycleOfLife(population):
    global generation;

    print("START CYCLE OF LIFE")

    while (not termination(population)):
        generation += 1;

        parents = parentSelection(population, 4);
        children = crossover(parents);
        children = mutation(children);
        children = childSelection(children, 2);

        population = parents + children;

        population.sort(reverse = True, key=evaluatePopulation)

        maxIndividual = population[0];
        print("END OF GENERATION: ", generation)
        print(maxIndividual["value"]);
        print(maxIndividual["individual"]["generation"])
        print(maxIndividual["individual"]["chromosomes"])


    print("FINAL RESULT: ")
    maxIndividual = population[0];
    print(maxIndividual["value"]);
    print(maxIndividual["individual"]["generation"])
    print(maxIndividual["individual"]["chromosomes"])




#Termination condition
def termination(population):
    return generation >= MAX_ITERATIONS or len(population) == 0 or population[0]["value"] == 1;

#Parent selection algorithm
def parentSelection(population, numParents):
    return population[0:numParents];

#Child selection algorithm
def childSelection(population, numChildren):
    return population[0:numChildren];

#crossover algorithm
def crossover(population):
    newPopulation = [];

    for i in range(len(population)):
        if (i % 2 == 1):
            newPopulation.extend(uniformCrossover(
                population[i - 1]["individual"]["chromosomes"],
                population[i]["individual"]["chromosomes"]
            ));

    return newPopulation;

def uniformCrossover(parent1, parent2):
    child1 = {
        "generation": generation,
        "chromosomes": {
            "hid_lay": round(CROSSOVER_ALPHA * parent1["hid_lay"] + (1 - CROSSOVER_ALPHA) * parent2["hid_lay"]),
            "hid_node": round(CROSSOVER_ALPHA * parent1["hid_node"] + (1 - CROSSOVER_ALPHA) * parent2["hid_node"]),
            "lr": round(CROSSOVER_ALPHA * parent1["lr"] + (1 - CROSSOVER_ALPHA) * parent2["lr"], 3)
        }
    };
    child2 = {
        "generation": generation,
        "chromosomes": {
            "hid_lay": round(CROSSOVER_ALPHA * parent2["hid_lay"] + (1 - CROSSOVER_ALPHA) * parent1["hid_lay"]),
            "hid_node": round(CROSSOVER_ALPHA * parent2["hid_node"] + (1 - CROSSOVER_ALPHA) * parent1["hid_node"]),
            "lr": round(CROSSOVER_ALPHA * parent2["lr"] + (1 - CROSSOVER_ALPHA) * parent1["lr"], 3)
        }
    };

    return [{
        "individual": child1,
        "value": 0
    }, {
        "individual": child2,
        "value": 0
    }];

#mutation algorithm
def mutation(population):

    for i in range(len(population)):
        if (random.random() < MUTATION_PROPABILITY):
            population[i]["individual"]["chromosomes"]["hid_lay"] = round(
                np.random.uniform(HID_LAY_LIMITS[0], HID_LAY_LIMITS[1])
            )
        if (random.random() < MUTATION_PROPABILITY):
            population[i]["individual"]["chromosomes"]["hid_node"] = round(
                np.random.uniform(HID_NODE_LIMITS[0], HID_NODE_LIMITS[1])
            )
        if (random.random() < MUTATION_PROPABILITY):
            population[i]["individual"]["chromosomes"]["lr"] = round(
                np.random.uniform(LR_LIMITS[0], LR_LIMITS[1]), 3
            )

        population[i]["value"] = evaluateIndividual(population[i]["individual"])

    population.sort(reverse = True, key=evaluatePopulation)

    return population;

#This code will change according to the problem
def createIndividual(params):

    return {
        "generation": generation,
        "chromosomes": params
    };

def evaluatePopulation(individual):
    return individual["value"];

def evaluateIndividual(individual):

    hid_lay = individual["chromosomes"]["hid_lay"]
    hid_node = [individual["chromosomes"]["hid_node"]] * individual["chromosomes"]["hid_lay"]
    hid_act = [tf.nn.relu] * individual["chromosomes"]["hid_lay"]
    tf_opt = tf.keras.optimizers.SGD(lr=individual["chromosomes"]["lr"],
        momentum=0.0,
        decay=0.0,
        nesterov=False)

    tf_model = TensorFlowMNIST(DATA_PATH, hid_lay, hid_node, hid_act, TF_EP)
    loss, accuracy = tf_model.train(2)

    return accuracy;


main();
