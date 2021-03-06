import time, math, random, bisect, argparse
import gym
import numpy as np
import pickle

goodBytes =  [70, 99, 101] #, 57, 70, 71, 72, 74, 75, 86, 90, 91, 94, 95, 99, 101, 102, 103, 104, 105, 107, 109, 119, 121, 122]

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

actionMap = {
    0: 0,
    1: 2,
    2: 3
}

def run(env, network, display=False, save=False):
    if display or save:
        env.seed(network.lastSeed)
    else:
        network.lastSeed = env.seed()[0]

    if isinstance(env, gym.wrappers.Monitor):
        env.enabled = save
        if save:
            env.file_infix = '%04d' % network.fitness

    actions = set()

    observation = env.reset()

    result = 0
    while True:
        if display: env.render()

        input = observation[goodBytes] / 255
        output = network.getOutput(input)

        res, mx = 0, output[0]
        for idx, val in enumerate(output):
            if val > mx:
                res, mx = idx, val

        res = actionMap[res]

        if observation[101] == 0:
            res = 1

        actions.add(res)

        observation, reward, done, info = env.step(res)
        result += reward
        if done: break

    return result


def runSeries(env, network, display = False, save = False, count = 5, scoreExtractor = lambda x: sum(x) / len(x)):
    result = []
    for i in range(count):
        result.append(run(env, network, display = display, save = save))

    return scoreExtractor(result)


def main(args):
    env = gym.make('Breakout-ram-v0')
    env.unwrapped.frameskip = 1

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

        population.evolve()


    print("Final population:")
    for network in population.population:
        print(run(env, network))

    print("Mean fitness: %d" % (sum([x.fitness for x in population.population]) / args.size))

    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--display', action='store_true', default=False, 
        help="If specified, generations before final will be displayed")
    parser.add_argument('-s', '--size', type=int, default=30,
        help="Population size")
    parser.add_argument('-g', '--generations', type=int, default=100,
        help="Maximum generatoin count")
    parser.add_argument('-mr', '--mutation-rate', type=float, default=0.07,
        help="Mutation rate (0 .. 1)")
    parser.add_argument('-nc', '--node-count', type=int, nargs='+', default=Network.getOptimalNodeCount(len(goodBytes), 3),
        help="List of network layer sizes")
    parser.add_argument('-dir', type=str, default='/tmp/openai',
        help="Directory to save network dumps and replays")
    args = parser.parse_args()

    main(args)
