import time, math, random, bisect, argparse
import gym
import numpy as np
import pickle


class Network: 

    LAST_ID = 0
    MEAN_FITNESS = True
    MAX_FITNESS = False

    def __init__(self, count):     
        self.id = Network.LAST_ID
        Network.LAST_ID += 1

        self.lastSeed = 0
        self.count = count
        self.layers = len(count) - 1
        self._fitness = [0, 0]
        self.weights, self.biases = [], []
        for i in range(self.layers):
            self.weights.append(np.random.uniform(-1, 1, (count[i], count[i + 1])))
            self.biases.append(np.random.uniform(-1, 1, count[i + 1]))

    @property
    def fitness(self):
        if Network.MEAN_FITNESS:
            return self._fitness[0] / self._fitness[1] if self._fitness[1] > 0 else 0
        return self._fitness[0]

    @fitness.setter
    def fitness(self, value):
        if Network.MEAN_FITNESS:
            self._fitness[0] += value
            self._fitness[1] += 1
        elif Network.MAX_FITNESS:
            self._fitness[0] = max(self._fitness[0], value)
        else:
            self._fitness[0] = value

    def dump(self, filename):
        with open(filename, 'bw') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        with open(filename, 'br') as f:
            return pickle.load(f)

    def getOutput(self, input):
        output = input
        for i in range(self.layers):
            output = np.maximum(0, np.matmul(output, self.weights[i]) + self.biases[i])
        return output

    @staticmethod
    def getOptimalNodeCount(input, output):
        return [input, int(np.sqrt(input * output)), output]


    def print(self):
        return "%4d - %2d" % (int(self.fitness), self.id)


class Population:

    def __init__(self, size, mutationRate, nodeCount):
        self.nodeCount = nodeCount
        self.size = size
        self.mutationRate = mutationRate
        self.population = [Network(nodeCount) for i in range(size)]


    def sort(self):
        self.population.sort(key = lambda x: x.fitness, reverse = True)


    def evolve(self):
        self.sort()

        bestCount = max(1, int(np.sqrt(self.size)))
        basePopulation = self.population[:bestCount]
        self.population = list(basePopulation)

        for A in basePopulation:
            for B in basePopulation:
                if A != B:
                    self.population.append(self.createChild(A, B))

        for A in basePopulation:
            if len(self.population) >= self.size:
                break

            self.population.append(self.createChild(A, A))

    def createChild(self, networkA, networkB):     
        def smartDivision(a, b):
            return 0 if a == 0 else a / b

        def mutate(list, idx, itemA, itemB):
            if random.random() > self.mutationRate:
                first = random.random() < smartDivision(networkA.fitness, networkA.fitness + networkB.fitness)
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


def run(env, network, display=False, save=False):
    if display or save:
        env.seed(network.lastSeed)
    else:
        network.lastSeed = env.seed()[0]

    if isinstance(env, gym.wrappers.Monitor):
        env.enabled = save
        if save:
            env.file_infix = '%04d' % network.fitness

    observation = env.reset()

    result = 0
    while True:
        if display: env.render()

        output = network.getOutput(observation)
        output = [0, 1][sigmoid(output[0]) > 0.5]

        observation, reward, done, info = env.step(output)
        result += reward
        if done: break

    return result


def runSeries(env, network, display = False, save = False, count = 5, scoreExtractor = lambda x: sum(x) / len(x)):
    result = []
    for i in range(count):
        result.append(run(env, network, display = display, save = save))

    return scoreExtractor(result)


def main(args):
    env = gym.make('CartPole-v0')

    env.seed(0)


    env = gym.wrappers.Monitor(env, args.dir, force=True, video_callable = lambda x: True)

    population = Population(args.size, args.mutation_rate, args.node_count)

    for generation in range(args.generations):
        for idx, network in enumerate(population.population):
            network.fitness = runSeries(env, network, display = args.display)

            print("Generation %4d Sample %3d -> Fitness %7s" % (generation, idx, network.print()))
        
        population.sort()
        print([x.print() for x in population.population])

        net = population.population[0];
        run(env, net, save = True)
        net.dump(args.dir + '/openaigym-dump-%d-%d' % (net.id, net.fitness));


        min_fitness = min([x.fitness for x in population.population])
        if min_fitness == 200: break


        population.evolve()


    print("Final population:")
    for network in population.population:
        print(run(env, network))

    print("Mean fitness: %d" % (sum([x.fitness for x in population.population]) / args.size))

    population.sort()
    print("Running random games with best sample")

    bestNetwork = population.population[0]
    for i in range(100):
        print("%5d -> %3d" % (i, run(env, bestNetwork, display = True, max_steps = args.max_steps)))


    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--display', action='store_true', default=False, 
        help="If specified, generations before final will be displayed")
    parser.add_argument('-s', '--size', type=int, default=30,
        help="Population size")
    parser.add_argument('-g', '--generations', type=int, default=100,
        help="Maximum generatoin count")
    parser.add_argument('-mr', '--mutation-rate', type=float, default=0.01,
        help="Mutation rate (0 .. 1)")
    parser.add_argument('-nc', '--node-count', type=int, nargs='+', default=Network.getOptimalNodeCount(4, 1),
        help="List of network layer sizes")
    parser.add_argument('-dir', type=str, default='/tmp/openai/cartpole',
        help="Directory to save network dumps and replays")
    args = parser.parse_args()

    main(args)