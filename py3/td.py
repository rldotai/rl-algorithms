"""
Temporal difference learning, AKA TD(λ), implemented in Python 3.

This is one of the foundational algorithms of reinforcement learning.
See the book "Reinforcement Learning: An Introduction" by Sutton and Barto for
a full introduction, in particular Chapter 7.

The algorithm is given in pseudocode on Rich's website[0].

This version of TD(λ) is an on-policy algorithm, so it doesn't respond
well to updates from trajectories generated via policies other than the one
it is currently evaluating.
There are a slew of modifications that can allow for off-policy evaluation,
for example: GTD(λ), ETD(λ), and other importance sampling methods.
Here, we employ accumulating traces (vs. replacing traces or dutch traces),
although modifying the code for different traces is straightforward.


---

0: https://webdocs.cs.ualberta.ca/~sutton/book/ebook/node75.html
"""
import numpy as np


class TD:
    """Temporal Difference Learning or TD(λ).

    Attributes
    ----------
    n : int
        The number of features (and therefore the length of the weight vector).
    z : Vector[float]
        The eligibility trace vector.
    w : Vector[float]
        The weight vector.
    """
    def __init__(self, n):
        """Initialize the learning algorithm.

        Parameters
        -----------
        n : int
            The number of features, i.e. expected length of the feature vector.
        """
        self.n = n
        self.w = np.zeros(self.n)
        self.z = np.zeros(self.n)

    def get_value(self, x):
        """Get the approximate value for feature vector `x`."""
        return np.dot(self.w, x)

    def update(self, x, r, xp, alpha, gm, gm_p, lm):
        """Update from new experience, i.e. from a transition `(x,r,xp)`.


        Parameters
        ----------
        x : array_like
            The observation/features from the current timestep.
        r : float
            The reward from the transition.
        xp : array_like
            The observation/features from the next timestep.
        alpha : float
            The step-size parameter for updating the weight vector.
        gm : float
            Gamma, abbreviated `gm`, the discount factor for the current state.
        gm_p : float
            The discount factor for the next state.
        lm : float
            Lambda, abbreviated `lm`, is the bootstrapping parameter for the
            current timestep.

        Notes
        -----
        Features (`x` and `xp`) are assumed to be 1D arrays of length `self.n`.
        Other parameters are floats but are generally expected to be in the
        interval [0, 1].
        """
        delta = r + gm_p*np.dot(self.w, xp) - np.dot(self.w, x)
        self.z = x + gm*lm*self.z
        self.w += alpha*delta*self.z

    def reset(self):
        """Reset weights, traces, and other parameters."""
        self.w = np.zeros(self.n)
        self.z = np.zeros(self.n)
