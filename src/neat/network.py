import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class NeuralNetwork:
    def __init__(self, genome):
        self.genome = genome

    def forward(self, inputs: np.ndarray) -> float:
        values = {}

        # Initialize input nodes
        input_nodes = [n for n in self.genome.nodes.values() if n.type == "input"]
        for node, val in zip(input_nodes, inputs):
            values[node.id] = val

        # Initialize other nodes
        for node in self.genome.nodes.values():
            if node.type != "input":
                values[node.id] = node.bias

        # Feedforward
        for conn in sorted(self.genome.connections.values(), key=lambda c: c.innovation):
            if not conn.enabled:
                continue
            values[conn.out_node] += values[conn.in_node] * conn.weight

        # Output node
        output_node = max(
            (n for n in self.genome.nodes.values() if n.type == "output"),
            key=lambda n: n.id,
        )
        return sigmoid(values[output_node.id])
