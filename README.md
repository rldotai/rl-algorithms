# rl-algorithms

Reinforcement learning algorithms.

There are many different variants on the basic ideas of reinforcement learning.
I have implemented some of them, with a focus on linear function approximation.

Extending these algorithms (for example, with nonlinear function approximators such as neural nets) is relatively straightforward once you are familiar with the underlying ideas. 

## Implemented Algorithms

- [Temporal Difference Learning, TD(λ)](py3/td.py)
- [Least-Squares Temporal Difference Learning, LSTD(λ)](py3/lstd.py)
- [Emphatic Temporal Difference Learning, ETD(λ)](py3/etd.py)
- [Gradient Temporal Difference Learning, GTD(λ), AKA TDC(λ)](py3/gtd.py)
- [True-Online Temporal Difference Learning, AKA TD with "Dutch Traces"](py3/totd.py)
- [Least Squares Emphatic Temporal Difference Learning, ELSTD(λ)](py3/elstd.py)
- [Hybrid Temporal Difference Learning](py3/htd.py)

# Contributing

Send me a pull request if you have code to contribute.

Alternatively, raise an issue and provide me with a link to the paper describing the algorithm, and I will read and implement it when I get a chance. 
