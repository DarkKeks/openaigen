import time, math, random, bisect, argparse
import gym
import numpy as np


class Network: 

    def __init__(self, count):     
        self.count = count
        self.layers = len(count) - 1
        self._fitness = [0, 0]
        self.weights, self.biases = [], []
        for i in range(self.layers):
            self.weights.append(np.random.uniform(-1, 1, (count[i], count[i + 1])))
            self.biases.append(np.random.uniform(-1, 1, count[i + 1]))

    @property
    def fitness(self):
        return self._fitness[0] / self._fitness[1] if self._fitness[1] > 0 else 0

    @fitness.setter
    def fitness(self, value):
        self._fitness[0] += value
        self._fitness[1] += 1

    def getOutput(self, input):
        output = input
        for i in range(self.layers):
            output = np.tanh(np.matmul(output, self.weights[i]) + self.biases[i])
        return output


class Population:

    def __init__(self, size, survivalRate, mutationRate, nodeCount):
        self.nodeCount = nodeCount
        self.size = size
        self.mutationRate = mutationRate
        self.survivalRate = survivalRate
        self.population = [Network(nodeCount) for i in range(size)]


    def sort(self):
        self.population.sort(key = lambda x: x.fitness, reverse = True)


    def evolve(self):
        self.sort()

        bestCount = min(1, int(self.survivalRate * self.size))
        self.population = self.population[:bestCount]

        fitness = [0]
        baseFitness = min([x.fitness for x in self.population])
        for i, network in enumerate(self.population):
            fitness.append(fitness[i] + (network.fitness - baseFitness) ** 4)

        def findNearest(array, value):
            previous = 0.0

            for idx, current in enumerate(array):
                if (value >= previous) and (value <= current):
                    return idx - 1 if ((value - previous) < (current - value)) and (idx > 0) else idx
                previous = current


        def getRandomNetwork():
            value = random.uniform(0, fitness[-1])
            return findNearest(fitness, value)#bisect.bisect_left(fitness, value)

        while len(self.population) < self.size:
            idx = [getRandomNetwork() for i in range(2)]
            self.population.append(self.createChild(*[self.population[i] for i in idx]))


    def createChild(self, networkA, networkB):        

        def mutate(list, idx, itemA, itemB):
            if random.random() > self.mutationRate:
                first = random.random() < networkA.fitness / (networkA.fitness + networkB.fitness)
                list[idx] = itemA if first else itemB

        result = Network(self.nodeCount)
        for i in range(len(result.weights)):
            for j in range(len(result.weights[i])):
                for k in range(len(result.weights[i][j])):
                    mutate(result.weights[i][j], k, 
                        networkA.weights[i][j][k],
                        networkB.weights[i][j][k])

        for i in range(len(result.biases)):
            for j in range(len(result.biases[i])):
                mutate(result.biases[i], j,
                    networkA.biases[i][j],
                    networkB.biases[i][j])

        return result


def sigmoid(x): 
    return 1.0 / (1.0 + np.exp(-x))


def run(env, network, max_steps=100000, display=True):
    observation = env.reset()

    result = 0
    for step in range(max_steps):
        if display:
            env.render()

        output = network.getOutput(observation / 255.0)

        res, mx = 0, output[0]
        for idx, val in enumerate(output):
            if val > mx:
                res = idx
                mx = val

        observation, reward, done, info = env.step(res)
        result += reward
        if done: break

    return result


def main(args):
    env = gym.make('Breakout-ram-v0')

    env.seed(0)

    population = Population(args.size, args.survival_rate, args.mutation_rate, args.node_count)

    for generation in range(args.generations):
        for idx, network in enumerate(population.population):
            network.fitness = run(env, network, max_steps = args.max_steps, display = args.display)

            print("Generation %4d Sample %3d -> Fitness %4d" % (generation, idx, network.fitness))

        population.evolve()

    print("Final population:")
    for network in population.population:
        print(run(env, network, max_steps = args.max_steps))

    print("Mean fitness: %d" % (sum([x.fitness for x in population.population]) / args.size))

    population.sort()
    print("Running random games with best sample")

    bestNetwork = population.population[0]
    for i in range(100):
        print("%5d -> %3d" % (i, run(env, bestNetwork, max_steps = args.max_steps)))


    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--display', action='store_true', default=False, 
        help="If specified, generations before final will be displayed")
    parser.add_argument('-s', '--size', type=int, default=20,
        help="Population size")
    parser.add_argument('-g', '--generations', type=int, default=100,
        help="Maximum generatoin count")
    parser.add_argument('-mr', '--mutation-rate', type=float, default=0.1,
        help="Mutation rate (0 .. 1)")
    parser.add_argument('-sr', '--survival-rate', type=float, default=0.2,
        help="Survival rate (0 .. 1)")
    parser.add_argument('-nc', '--node-count', type=int, nargs='+', default=[128, 100, 80, 60, 20, 4],
        help="List of network layer sizes")
    parser.add_argument('-ms', '--max-steps', type=int, default=200,
        help="Maximum game steps")

    args = parser.parse_args()

    print(args)


    main(args)
