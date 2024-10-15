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
