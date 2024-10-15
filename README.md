# Jupyter Notebook for the 2024 Nobel Prize in Physics

The 2024 Nobel Prize in Physics was awarded to
Prof. John J. Hopfield (Princeton University) and
Prof. Geoffrey E. Hinton (University of Toronto)
"for foundational discoveries and inventions that enable machine
learning with artificial neural networks."

This repository contains a Jupyter notebook to help students explore
and understand the scientific achievements behind this prize through
interactive, hands-on learning.
The target audience is advanced undergraduate and graduate students in
the fields of physical science and computer science.

The Jupyter notebook is synced to a markdown file using `jupytext`.
If you clone this repository from GitHub, please run
```
python3 -m venv venv
. venv/bin/activate
pip install -r requirements.txt
jupytext --sync notebook.md
jupyter lab notebook.ipynb
```
to access the notebook.
