# rl-algorithms

Reinforcement learning algorithms.

There are many different variants on the basic ideas of reinforcement learning.
I have implemented some of them, with a focus on linear function approximation.

Extending these algorithms (for example, with nonlinear function approximators such as neural nets) is relatively straightforward once you are familiar with the underlying ideas.

To facilitate this, the algorithms listed are written in a straightforward style and thoroughly commented, with references to the relevant papers and some explanation of the reasoning behind the code.

## Implemented Algorithms

- [TD(λ): Temporal Difference Learning](py3/td.py)
- [LSTD(λ): Least-Squares Temporal Difference Learning](py3/lstd.py)
- [ETD(λ): Emphatic Temporal Difference Learning](py3/etd.py)
- [GTD(λ): Gradient Temporal Difference Learning, AKA TDC(λ)](py3/gtd.py)
- [TOTD(λ): True-Online Temporal Difference Learning, AKA TD with "Dutch Traces"](py3/totd.py)
- [ESTD(λ): Least Squares Emphatic Temporal Difference Learning](py3/elstd.py)
- [HTD(λ): Hybrid Temporal Difference Learning](py3/htd.py)
- [DTD(λ) or TD-δ^2: Online Variance Estimation via temporal difference errors](py3/td-variance.py)

# Contributing

Send me a pull request if you have code to contribute.

Alternatively, raise an issue and provide me with a link to the paper describing the algorithm, and I will read and implement it when I get a chance.
