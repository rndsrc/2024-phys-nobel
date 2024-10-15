---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.4
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Jupyter Notebook for the 2024 Nobel Prize in Physics


## Historical Introduction to the Ising Model

The Ising model is a mathematical model in statistical mechanics, introduced by Wilhelm Lenz in 1920 and solved for the one-dimensional case by his student Ernst Ising in 1925. The model was originally used to explain ferromagnetism, where magnetic materials exhibit spontaneous magnetization due to interactions between neighboring atomic spins.

* Wilhelm Lenz conceived the model as a simplified representation of magnetic interactions in a lattice, where spins can either point “up” (+1) or “down” (-1).
* Ernst Ising solved the one-dimensional version of the model in his doctoral thesis, showing that it did not exhibit phase transitions—a result that was surprising at the time.
* Lars Onsager solved the two-dimensional version of the model in 1944, demonstrating that it undergoes a phase transition at a critical temperature, where spontaneous magnetization occurs.

Since then, the Ising model has become one of the most widely studied models in statistical physics and beyond. Its applications extend not only to physics (such as in magnetism and lattice gases) but also to fields like biology (neural networks), computer science (optimization problems), and even sociology (modeling opinion dynamics).

The Metropolis algorithm (1953) was developed for Monte Carlo simulations of systems like the Ising model, enabling the study of large, complex systems by simulating their thermal fluctuations and statistical properties. This method revolutionized computational physics and remains a powerful tool in many areas of research today.

In this hands-on lab, we will implement the Ising model using the Metropolis algorithm in Python.


## An Implementation of an Ising Model

The Ising model is pretty simple so we only need to use two packages, `numpy` for array handing and `matplotlib` for plotting.

```python
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
```

For easy comparison of simulation paramaters, we will implement the Ising model as a class.

```python
class IsingModel:

    def __init__(self, T, shape=(64,64)):
        """Initialize an Ising model with temperature `T` and grid shape `shape`"""
        self.T = T
        self.grid = np.random.choice([-1,1], size=shape)
        self.magnetization = [np.sum(self.grid)]

    def dE(self, i, j):
        I,J = self.grid.shape
        return 2 * self.grid[i,j] * (
            self.grid[ i     ,(j-1)%J] + self.grid[ i     ,(j+1)%J] +
            self.grid[(i-1)%I, j     ] + self.grid[(i+1)%I, j     ]
        )

    def step(self):
        i  = np.random.randint(0, self.grid.shape[0])
        j  = np.random.randint(0, self.grid.shape[1])
        dE = self.dE(i,j)
        m  = self.magnetization[-1]
        if dE < 0 or np.random.rand() < np.exp(-dE / self.T):
            self.grid[i,j] *= -1
            m += 2 * self.grid[i,j]
        self.magnetization.append(m)

    def run(self, N):
        for n in range(N):
            self.step()

    def plot(self):
        fig, (ax0, ax1) = plt.subplots(1,2, figsize=(12,6))
        ax0.imshow(I.grid)
        ax1.plot(np.array(I.magnetization) / I.grid.size)
```

```python
I = IsingModel(2)
I.run(64*64*100)
I.plot()
```

## Hopfield Network

The Hopfield network, developed by John Hopfield in 1982, is an artificial neural network model with a deep connection to the Ising model.
This connection is key to understanding why Hopfield's work, recognized in the 2024 Nobel Prize in Physics, has had such a profound impact on the fields of neural networks and machine learning.

At its core, the Hopfield network functions as a recurrent neural network designed to store and recall patterns.
Each "neuron" in the network can be in one of two states: active (+1) or inactive (-1).
This binary nature closely mirrors the spin states in the Ising model, where each spin is either "up" or "down".
The neurons are interconnected, and the network evolves by updating the state of each neuron based on the states of its neighbors, following rules that minimize the system's overall energy.

What makes this particularly interesting is that the energy minimization process in the Hopfield network is mathematically analogous to how the Ising model works.
In both systems, there is a well-defined energy function that describes the interaction between units---spins in the Ising model, and neurons in the Hopfield network.
The system naturally evolves toward a state that minimizes this energy, and for the Hopfield network, these low-energy states correspond to stored memory patterns.
When a noisy or incomplete input is presented to the network, it "relaxes" into one of these low-energy states, effectively recalling the stored pattern.

This shared framework of energy minimization is what connects the Hopfield network so closely to the Ising model.
In fact, the mathematical structure of the energy function in a Hopfield network is very similar to the Hamiltonian of the Ising model, where the weights between neurons play a role analogous to the coupling between spins.

The recognition of the Hopfield network in the 2024 Nobel Prize highlights this elegant crossover between physics and neural computation.
The ideas from statistical mechanics, particularly the energy minimization concepts of the Ising model, laid the groundwork for significant advances in understanding how networks of simple elements---whether spins or neurons---can produce complex, emergent behavior.

```python
class HopfieldNetwork:
    def __init__(self, shape=(64,64)):
        size = np.prod(shape)
        self.shape   = shape
        self.weights = np.zeros((size, size))

    def train(self, patterns):
        for p in patterns:
            p = np.reshape(p, (self.weights.shape[0], 1)) # reshape the pattern to a column vector
            self.weights += np.dot(p, p.T)                # Hebbian learning rule: W += p * p.T
        np.fill_diagonal(self.weights, 0) # ensure no neuron connects to itself
        self.weights /= len(patterns)     # normalize by the number of patterns

    def init(self):
        self.grid = np.random.choice([-1,1], size=self.shape)

    def step(self):
        i = np.random.randint(0, self.shape[0])
        j = np.random.randint(0, self.shape[1])
        self.grid[i,j] = np.sign(np.dot(self.weights[i*self.shape[1]+j,:], self.grid.flatten()))

    def run(self, N):
        for n in range(N):
            self.step()
```

```python
with Image.open("A.png") as f:
    A = (np.array(f)[::4,::4,0] >= 128).astype(int) * 2 - 1

with Image.open("C.png") as f:
    C = (np.array(f)[::4,::4,0] >= 128).astype(int) * 2 - 1

fig, (ax0, ax1) = plt.subplots(1,2)
ax0.imshow(A)
ax1.imshow(C)
```

```python
h = HopfieldNetwork()
h.train([A,C])
```

```python
h.init()
h.run(256*256)
plt.imshow(h.grid)
```

```python
h.init()
h.run(256*256)
plt.imshow(h.grid)
```

```python

```
