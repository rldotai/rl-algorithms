"""
Emphatic Temporal Difference Learning Algorithm (ETD), implemented in Python 3.
"""
import numpy as np 


class ETD:
    """Emphatic Temporal Difference Learning, or ETD(λ).

    Attributes
    ----------
    n : int
        The number of features (and therefore the length of the weight vector).
    z : Vector[float]
        The eligibility trace vector.
    theta : Vector[float]
        The weight vector.
    F : float
        The followon trace scalar.
    M : float
        The emphasis scalar.
    """
    def __init__(self, n):
        """Initialize the learning algorithm.

        Parameters
        -----------
        n : int
            The number of features
        """
        self.n = n
        self.theta = np.zeros(self.n)
        self.z = np.zeros(self.n)
        self.F = 0
        self.M = 0

    def update(self, x, r, xp, alpha, gm, gm_p, lm, rho, interest):
        """Update from new experience.

        
        Parameters
        ----------
        x : array_like
            The observation/features from the current timestep.
        r : float
            The reward from the transition.
        xp : array_like
            The observation/features from the next timestep.
        alpha : float
            The stepsize parameter for the update.
        gm : float 
            Gamma, abbreviated `gm`, the discount factor for the current state.
        gm_p : float 
            The discount factor for the next state.
        lm : float 
            Lambda, abbreviated `lm`, is the bootstrapping parameter for the 
            current timestep.
        rho : float 
            The importance sampling ratio between the target policy and the 
            behavior policy for the current timestep.
        interest : float 
            The interest for the current timestep.

        Notes
        -----
        Features (`x` and `xp`) are assumed to be 1D arrays of length `self.n`.
        Other parameters are floats but are generally expected to be in the 
        interval [0, 1].
        """
        delta = r + gm_p*np.dot(self.theta, xp) - np.dot(self.theta, x)
        self.F = gm*self.F + interest
        self.M = lm*interest + (1 - lm)*self.F
        self.z = rho*(x*self.M + gm*lm*self.z)
        self.theta += alpha*delta*self.z

        # prepare for next iteration
        self.F *= rho

    def reset(self):
        """Reset weights, traces, and other parameters."""
        self.F = 0
        self.M = 0
        self.theta = np.zeros(self.n)
        self.z = np.zeros(self.n)