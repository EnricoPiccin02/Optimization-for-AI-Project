import random
from src.neat.genome import Genome, NodeGene, ConnectionGene
from src.neat.fitness import evaluate_genome


class Population:
    def __init__(self, config, mean, std, baseline):
        self.config = config
        self.mean = mean
        self.std = std
        self.baseline = baseline
        self.innovation = 0

        self.genomes = [
            self._initial_genome() for _ in range(self.config["population_size"])
        ]

        self.best_genome = None

    def _initial_genome(self):
        g = Genome()
        num_inputs = self.config["num_inputs"]

        # Input nodes
        for i in range(num_inputs):
            g.nodes[i] = NodeGene(i, "input")

        # Single output node
        output_id = num_inputs
        g.nodes[output_id] = NodeGene(output_id, "output")

        # Fully connect inputs to output
        for i in range(num_inputs):
            g.connections[self.innovation] = ConnectionGene(
                in_node=i,
                out_node=output_id,
                weight=random.uniform(-1.0, 1.0),
                enabled=True,
                innovation=self.innovation,
            )
            self.innovation += 1

        return g

    def evaluate(self):
        for g in self.genomes:
            g.fitness = evaluate_genome(
                g,
                self.mean,
                self.std,
                self.baseline,
                self.config["games_per_eval"],
            )

        self.best_genome = max(self.genomes, key=lambda g: g.fitness)

    def reproduce(self):
        self.genomes.sort(key=lambda g: g.fitness, reverse=True)
        survivors = self.genomes[: len(self.genomes) // 2]

        new_genomes = [g.copy() for g in survivors]

        while len(new_genomes) < self.config["population_size"]:
            if random.random() < self.config["crossover_prob"]:
                p1, p2 = random.sample(survivors, 2)
                child = p1.crossover(p2)
            else:
                parent = random.choice(survivors)
                child = parent.copy()

            self._mutate(child)
            new_genomes.append(child)

        self.genomes = new_genomes

    def _mutate(self, genome):
        for conn in genome.connections.values():
            if random.random() < self.config["weight_mutation_prob"]:
                conn.weight += random.gauss(0.0, self.config["weight_mutation_sigma"])
