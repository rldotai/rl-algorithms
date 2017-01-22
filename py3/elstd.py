"""
Emphatic least-squares temporal difference learning, also known as ELSTD(λ).

TODO: Test the implementation
TODO: Add documentation
TODO: Add citations
"""
import numpy as np 


class ELSTD:
    """Emphatic least-squares temporal difference learning.

    Attributes
    ----------
    n : int
        The number of features (and therefore the length of the weight vector).
    z : Vector[float]
        The eligibility trace vector.
    A : Matrix[float]
        A matrix with shape `(n, n)` that acts like a potential matrix.
    b : Vector[float]
        A vector of length `n` that accumulates the trace multiplied by the
        reward over a trajectory.
    F : float
        The followon trace scalar.
    M : float
        The emphasis scalar.
    """
    def __init__(self, num_features=None, epsilon=0):
        """Initialize the learning algorithm.

        Parameters
        -----------
        n : int
            The number of features
        epsilon : float
            To avoid having the `A` matrix be singular, it is sometimes helpful
            to initialize it with the identity matrix multiplied by `epsilon`.
        """
        self.n = n
        self.reset(epsilon)
    
    def reset(self, epsilon=0):
        """Reset weights, traces, and other parameters."""
        self.z = np.zeros(self.n)
        self.A = np.eye(self.n) * epsilon
        self.b = np.zeros(self.n)
        self.F = 0
        self.M = 0

    @property
    def theta(self):
        """Compute the weight vector via `A^{-1} b`."""
        _theta = np.dot(np.linalg.pinv(self.A), self.b)
        return _theta

    def update(self, x, reward, xp, gm, gm_p, lm, interest):
        """Update from new experience, i.e. from a transition `(x,r,xp)`.

        Parameters
        ----------
        x : array_like
            The observation/features from the current timestep.
        r : float
            The reward from the transition.
        xp : array_like
            The observation/features from the next timestep.
        gm : float
            Gamma, abbreviated `gm`, the discount factor for the current state.
        gm_p : float
            The discount factor for the next state.
        lm : float
            Lambda, abbreviated `lm`, is the bootstrapping parameter for the
            current timestep.
        interest : float 
            The 'interest' in the current state, from which emphasis is derived.
        """
        self.F = gm * self.F + interest
        self.M = (lm * I) + ((1 - lm) * self.F)
        self.z = (gm * lm * self.z + self.M * x)
        self.A += np.outer(self.z, (x - gm_p*xp))
        self.b += self.z * reward