import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


class Neuron:
    def __init__(self, n_inputs, bias=0., weights=None):
        self.b = bias
        if weights is not None:
            self.ws = np.array(weights)
        else:
            self.ws = np.random.rand(n_inputs)

    def _f(self, x):
        return max(x * .1, x)

    def _f_derivative(self, x):
        return 0.1 if x < 0 else 1

    def __call__(self, xs):
        self.xs = xs
        self.z = xs @ self.ws + self.b
        self.a = self._f(self.z)
        return self.a


class NeuralLayer:
    def __init__(self, n_inputs, n_neurons):
        self.neurons = [Neuron(n_inputs) for _ in range(n_neurons)]

    def __call__(self, xs):
        self.inputs = xs
        self.outputs = np.array([neuron(xs) for neuron in self.neurons])
        return self.outputs

    def backprop(self, grad_outputs, learning_rate):
        grad_inputs = np.zeros_like(self.inputs)
        for i, neuron in enumerate(self.neurons):
            grad_z = grad_outputs[i] * neuron._f_derivative(neuron.z)
            grad_inputs += grad_z * neuron.ws
            neuron.ws -= learning_rate * grad_z * neuron.xs
            neuron.b -= learning_rate * grad_z
        return grad_inputs


class NeuralNetwork:
    def __init__(self, structure):
        self.layers = []
        for i in range(len(structure) - 1):
            self.layers.append(NeuralLayer(structure[i], structure[i + 1]))

    def __call__(self, xs):
        for layer in self.layers:
            xs = layer(xs)
        return xs

    def backprop(self, xs, ys, learning_rate):
        # Forward pass
        predictions = self(xs)
        # Compute loss (mean squared error)
        loss = np.mean((predictions - ys) ** 2)
        # Compute initial gradient
        grad_loss = 2 * (predictions - ys) / ys.size
        grad_outputs = grad_loss

        for layer in reversed(self.layers):
            grad_outputs = layer.backprop(grad_outputs, learning_rate)

        return loss


network_structure = [3, 4, 4, 1]
network = NeuralNetwork(network_structure)

# Visualization function
def visualize_network(structure):
    G = nx.DiGraph()
    subset_keys = {}
    for i, layer_size in enumerate(structure):
        for j in range(layer_size):
            neuron_name = f'Layer {i + 1} - Neuron {j + 1}'
            G.add_node(neuron_name)
            subset_keys[neuron_name] = i
        if i > 0:
            previous_layer_size = structure[i - 1]
            for prev_j in range(previous_layer_size):
                for curr_j in range(layer_size):
                    prev_neuron = f'Layer {i} - Neuron {prev_j + 1}'
                    curr_neuron = f'Layer {i + 1} - Neuron {curr_j + 1}'
                    G.add_edge(prev_neuron, curr_neuron)

    subsets = {}
    for node, subset in subset_keys.items():
        if subset not in subsets:
            subsets[subset] = []
        subsets[subset].append(node)

    pos = nx.multipartite_layout(G, subset_key=subsets)
    nx.draw(G, pos, with_labels=True)
    plt.show()


visualize_network(network_structure)

X = np.array([0.1, 0.2, 0.3])
Y = np.array([0.5])
learning_rate = 0.01

losses = []
for epoch in range(1000):
    loss = network.backprop(X, Y, learning_rate)
    losses.append(loss)
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss}')

# Plotting the loss over epochs
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
