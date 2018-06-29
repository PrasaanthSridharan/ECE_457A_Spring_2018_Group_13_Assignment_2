import heapq
import random

class TabuSearch:


    def __init__(self, tenure = 0, lower_tenure = None, upper_tenure = None, tenure_iteration = None,
                 candidatesNumber = 1, maxIterations = 0, neighborhood_probability = 1, max_repetition = None,
                 cost = None, initialSolution = []):

        self.matrix = [[0 for x in range(len(initialSolution))] for y in range(len(initialSolution))]
        self.lower_tenure = lower_tenure
        self.upper_tenure = upper_tenure
        self.tenure_iteration = tenure_iteration
        if self.lower_tenure is None or self.upper_tenure is None:
            self.tenure = tenure
        else:
            self.tenure = random.randint(self.lower_tenure, self.upper_tenure)
        self.cost = cost
        self.maxIterations = maxIterations
        self.solution = initialSolution.copy()
        self.value = self.cost(self.solution)
        self.candidatesNumber = candidatesNumber
        self.neighborhood_probability = neighborhood_probability
        self.max_repetition = max_repetition

        self.bestSolution = self.solution.copy()
        self.bestValue = self.value


    def selectCandidate(self, candidates):
        selectedCandidate = None
        for candidate in candidates:
            swap = candidate["swap"]
            if ((self.matrix[swap[0]][swap[1]] == 0
                and (self.max_repetition is None
                or self.matrix[swap[1]][swap[0]] < self.max_repetition))
                or candidate["cost"] < self.bestValue):
                selectedCandidate = candidate.copy()
                break

        for x in range(len(self.solution)):
            for y in range(x + 1, len(self.solution)):
                if self.matrix[x][y] > 0:
                    self.matrix[x][y] -= 1

        self.matrix[swap[0]][swap[1]] = self.tenure
        self.matrix[swap[1]][swap[0]] += 1

        self.solution[swap[0]], self.solution[swap[1]] = self.solution[swap[1]], self.solution[swap[0]]
        newValue = self.cost(self.solution)
        self.value = newValue

    def candidateCost(self, candidate):
        return candidate["cost"]

    def getCandidates(self):

        candidates = []

        for i in range(len(self.solution)):
            for j in range(i + 1, len(self.solution)):
                if random.random() < self.neighborhood_probability:
                    tempSolution = self.solution.copy()
                    tempSolution[i], tempSolution[j] = tempSolution[j], tempSolution[i]
                    cost = self.cost(tempSolution)
                    candidates.append({
                        "swap": sorted([i, j]),
                        "cost": cost
                    })

        return heapq.nsmallest(self.candidatesNumber, candidates, key=self.candidateCost)

    def search(self):
        iteration = 0

        while iteration < self.maxIterations:

            if self.tenure_iteration is not None and iteration % self.tenure_iteration == 0:
                self.tenure = random.randint(self.lower_tenure, self.upper_tenure)

            candidates = self.getCandidates()
            self.selectCandidate(candidates)

            if self.value < self.bestValue:
                self.bestSolution, self.bestValue = self.solution.copy(), self.value

            iteration += 1

        print("Best Solution: ", self.bestSolution)
        print("Best value: ", self.bestValue)
        return self.bestSolution
