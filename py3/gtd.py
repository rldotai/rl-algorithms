"""
Gradient-TD(λ) Learning Algorithm, via Adam White's doctoral thesis, pg. 47., 
and Maei's doctoral thesis pg. 74 and 91-92 for the original derivation and 
analysis. 
Note that the algorithm is referred to as TDC(λ) in that work, whereas GTD and 
GTD2 refer to variations on the same idea but without eligibility traces.

The advantage of GTD(λ) is its stability in the off-policy setting, at the 
expense of worse sample efficiency and therefore slower learning.

In Latex, the update equations look like:

δ_{t}   = R_{t+1} + γ_{t+1} w_{t}^T x_{t+1} - w_{t}^{T} x_{t} 
e_{t}   = ρ_{t} (λ_{t} γ_{t} e_{t-1} + x_{t})
w_{t+1} = w_{t} + α[ δ_{t} e_{t} + γ_{t+1} (1 - λ_{t}) ( e_{t}^{T} h_{t} ) x_{t+1} ]
h_{t+1} = h_{t} + β[ δ_{t} e_{t} - ( h_{t}^{T} x_{t} ) x_{t} ]

Where:
    - δ refers to the temporal difference error; 
    - γ is the discount parameter,
    - λ is the bootstrapping parameter
    - α and β are stepsizes parameters, 
    - w and h are weight vectors
    - e is the eligibility trace
    - x and r are feature vectors and rewards respectively.
"""
import numpy as np 


class GTD:
    """Gradient Temporal Difference Learning, or GTD(λ). Suitable for 
    off-policy learning, but with typically lower sample efficiency than TD(λ).

    Attributes
    ----------
    n : int
        The number of features (and therefore the length of the weight vector).
    e : Vector[float]
        The eligibility trace vector.
    w : Vector[float]
        The weight vector.
    h : Vector[float]
        The gradient adjustment weight vector.

    Notes
    -----
    See page 74 and 91-92 of Maei's thesis for definition of the algorithm.
    """
    def __init__(self, n):
        """Initialize the learning algorithm.

        Parameters
        -----------
        n : int
            The number of features, i.e. expected length of the feature vector.
        """
        self.n = n
        self.e = np.zeros(self.n)
        self.w = np.zeros(self.n)
        self.h = np.zeros(self.n)

    def get_value(self, x):
        """Get the approximate value for feature vector `x`."""
        return np.dot(self.w, x)

    def update(self, x, r, xp, alpha, beta, gm, gm_p, lm, rho:
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
            The stepsize parameter for updating the weight vector.
        beta : float 
            The stepsize parameter for updating the correction weights.
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

        Notes
        -----
        Features (`x` and `xp`) are assumed to be 1D arrays of length `self.n`.
        Other parameters are floats but are generally expected to be in the 
        interval [0, 1].
        """
        delta = r + gm_p*np.dot(self.theta, xp) - np.dot(self.theta, x)
        self.e = rho*(lm*gm*self.e + x)
        self.w += alpha*(delta*self.e + gm_p*(1-lm)*np.dot(self.e, self.h)*xp)
        self.h += beta*(delta*self.e + np.dot(self.h, x)*x)

    def reset(self):
        """Reset weights, traces, and other parameters."""
        self.e = np.zeros(self.n)
        self.w = np.zeros(self.n)
        self.h = np.zeros(self.n)

