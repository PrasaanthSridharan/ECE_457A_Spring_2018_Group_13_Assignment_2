from Tensorflow.tensorflow_mnist import TensorFlowMNIST

import sys
import random
import tensorflow as tf

#CONSTANTS
INERTIA = 0.792
C = 1.4944
R_1 = [random.random(), random.random(), random.random()]
R_2 = [random.random(), random.random(), random.random()]
LIMITS = [[1, 3], [8, 256], [0.001, 0.05]]
VARIABLES_PRECISION = [0, 0, 3]
MAX_ITERATIONS = 50
DATA_PATH = "MNIST_data/mnist.npz"
TF_EP = 3


def main():
    particles = []
    particles.append(createParticle(1, 256, 0.01))
    particles.append(createParticle(2, 16, 0.005))
    particles.append(createParticle(3, 128, 0.015))
    particles.append(createParticle(2, 8, 0.01))

    explore(particles)

def explore(particles):
    global_best_value = 0
    global_best_attributes = [0, 0, 0]

    cycle = 0

    while (cycle < MAX_ITERATIONS and global_best_value < 1):
        print("CYCLE: ", cycle)
        print("BEST VALUE: ", global_best_value)
        print("ATTRIBUTES: ", global_best_attributes)
        print(particles)
        for i in range(len(particles)):
            particles[i] = evaluateIndividual(particles[i])
            if particles[i]["value"] > global_best_value:
                global_best_value = particles[i]["value"]
                global_best_attributes = particles[i]["attributes"]
            if particles[i]["value"] > particles[i]["best_value"]:
                particles[i]["best_value"] = particles[i]["value"]
                particles[i]["best_position"] = particles[i]["attributes"]

            particleVelocity = particles[i]["velocity"]
            particlePosition = particles[i]["attributes"]

            for j in range(len(particleVelocity)):
                particleVelocity[j] = INERTIA * particleVelocity[j] + C * R_1[j] * (particles[i]["best_position"][j] - particlePosition[j]) + C * R_2[j] * (global_best_attributes[j] - particlePosition[j])

                newPosition = particlePosition[j] + particleVelocity[j]
                particlePosition[j] = round(min(LIMITS[j][1], max(LIMITS[j][0], newPosition)), VARIABLES_PRECISION[j])

            particles[i]["velocity"] = particleVelocity
            particles[i]["attributes"] = particlePosition


        cycle += 1

    print("************* FINISHED *************")
    print("BEST VALUE: ", global_best_value)
    print("ATTRIBUTES: ", global_best_attributes)

def createParticle(hid_lay, hid_node, lr):
    return {
        "value": None,
        "attributes": [hid_lay, hid_node, lr],
        "velocity": [0, 0, 0],
        "best_value": 0,
        "best_position": [0, 0, 0]
    }

def evaluateIndividual(individual):
    hid_lay = int(individual["attributes"][0])
    hid_node = [int(individual["attributes"][1])] * int(individual["attributes"][0])
    hid_act = [tf.nn.relu] * int(individual["attributes"][0])
    tf_opt = tf.keras.optimizers.SGD(lr=individual["attributes"][2],
        momentum=0.0,
        decay=0.0,
        nesterov=False)

    tf_model = TensorFlowMNIST(DATA_PATH, hid_lay, hid_node, hid_act, TF_EP)
    loss, accuracy = tf_model.train(2)

    individual["value"] = accuracy

    return individual;


main()
