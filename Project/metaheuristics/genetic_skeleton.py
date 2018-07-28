import sys;
import random;
import heapq;
from hashlib import sha256;


#CONSTANTS
MUTATION_PROPABILITY = 0.05;
CROSSOVER_PROBABILITY = 0.7;
POPULATION_SIZE = 40;
MAX_ITERATIONS = 1000000;

#GLOBALS
generation = 0;

def main():
    global generation;
    generation = 0;
    population = [];
    originalString = "test";

    for i in range(POPULATION_SIZE):
        population.append(createIndividual({
            "text": originalString + str(i)
        }));

    cycleOfLife(population);

# Contains all function calls for each cycle
def cycleOfLife(population):
    global generation;

    while (termination(population) is False):
        generation += 1;

        population = parentSelection(population);
        population = crossover(population);
        population = mutation(population);
        population = childSelection(population);


    maxIndividual = max(population, key=evaluateIndividual);
    print(evaluateIndividual(maxIndividual));
    print(maxIndividual["generation"])






#Termination condition
def termination(population):
    return generation >= MAX_ITERATIONS or len(population) == 0;

#Parent selection algorithm
def parentSelection(population):
    return population;

#Child selection algorithm
def childSelection(population):
    return heapq.nlargest(POPULATION_SIZE, population, key=evaluateIndividual);

def displayPopulation(population):
    for i in population:
        print(evaluateIndividual(i));

#crossover algorithm
def crossover(population):

    newPopulation = [];

    for i in range(len(population)):
        newPopulation.append(population[i]);

        if (i % 2 == 1 and random.random() < CROSSOVER_PROBABILITY):
            newPopulation.extend(onePointCrossover(population[i - 1]["chromosomes"], population[i]["chromosomes"]));

    return newPopulation;

def onePointCrossover(parent1, parent2):
    point = random.randint(1, len(parent1));
    child1 = {
        "generation": generation,
        "chromosomes": []
    };
    child2 = {
        "generation": generation,
        "chromosomes": []
    };

    for i in range(len(parent1)):
        if (i < point is True):
            child1["chromosomes"].append(parent1[i]);
            child2["chromosomes"].append(parent2[i]);
        else:
            child1["chromosomes"].append(parent2[i]);
            child2["chromosomes"].append(parent1[i]);

    return [child1, child2];

#mutation algorithm
def mutation(population):
    newPopulation = population;
    for i in range(len(population)):
        if (random.random() < MUTATION_PROPABILITY):
            value = population[i]["chromosomes"];
            newValue = [];

            for j in range(len(value)):
                if (random.random() < MUTATION_PROPABILITY):
                    if (value[j] == '0'):
                        newValue.append('1');
                    else:
                        newValue.append('0');
                else:
                    if (value[j] == '0'):
                        newValue.append('0');
                    else:
                        newValue.append('1');

            newPopulation.append({
                "generation": generation,
                "chromosomes": newValue
            });

    return newPopulation;

#This code will change according to the problem
def createIndividual(params):
    global generation;

    hash = sha256(str.encode(params["text"])).hexdigest();
    binary = list(bin(int(hash, 16))[2:].zfill(256));

    if ("generation" in params):
        generation = params["generation"];

    return {
        "generation": generation,
        "chromosomes": binary
    };

def evaluateIndividual(individual):
    return sha256(str.encode(''.join(individual["chromosomes"]))).hexdigest();


main();
