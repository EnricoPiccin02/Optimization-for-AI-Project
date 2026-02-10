from dataclasses import dataclass
import random
from typing import Dict


@dataclass
class NodeGene:
    id: int
    type: str  # 'input', 'hidden', 'output'
    bias: float = 0.0


@dataclass
class ConnectionGene:
    in_node: int
    out_node: int
    weight: float
    enabled: bool
    innovation: int


class Genome:
    def __init__(self):
        self.nodes: Dict[int, NodeGene] = {}
        self.connections: Dict[int, ConnectionGene] = {}
        self.fitness: float | None = None

    def copy(self):
        g = Genome()
        g.nodes = {k: NodeGene(**vars(v)) for k, v in self.nodes.items()}
        g.connections = {
            k: ConnectionGene(**vars(v)) for k, v in self.connections.items()
        }
        return g

    def crossover(self, other):
        """
        Uniform crossover assuming matching innovation numbers.
        """
        child = self.copy()

        for innov, conn in child.connections.items():
            if innov in other.connections and random.random() < 0.5:
                conn.weight = other.connections[innov].weight

        return child
