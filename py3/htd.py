"""
Hybrid-TD Learning Algorithm, via Adam White's doctoral thesis, pg. 173.

Like GTD(λ), it doesn't diverge in the off-policy case, while acting like TD(λ)
in the on-policy case, particularly with regards to good sample efficiency.

In Latex, the update equations look like:

δ_{t}   = R_{t+1} + γ_{t+1} w_{t}^T x_{t+1} - w_{t}^{T} x_{t} 
e_{t}   = ρ_{t} (λ_{t} γ_{t} e_{t-1} + x_{t})
z_{t}   = λ_{t} γ_{t} z_{t-1} + x_{t}
w_{t+1} = w_{t} + α[ δ_{t} e_{t} + (γ_{t+1} x_{t+1} - x_{t} ) (z_{t} - e_{t} ) h_{t} ]
h_{t+1} = h_{t} + β[ δ_{t} e_{t} + (γ_{t+1} x_{t+1} - x_{t} ) z_{t}^{T} h_{t}]

Where:
    - δ refers to the temporal difference error; 
    - γ is the discount parameter,
    - λ is the bootstrapping parameter
    - α and β are stepsize parameters, 
    - w and h are weight vectors
    - e and z are eligibility traces
    - x and r are feature vectors and rewards respectively.
"""
import numpy as np 


class HTD:
    """Hybrid Temporal Difference Learning, or HTD(λ).
    Acts like TD(λ) in the on-policy case, but with GTD(λ)'s stability when
    updating off-policy.

    Attributes
    ----------
    n : int
        The number of features (and therefore the length of the weight vector).
    e : Vector[float]
        The importance sampling eligibility trace vector.
    z : Vector[float]
        The on-policy eligibility trace vector.
    w : Vector[float]
        The weight vector.
    h : Vector[float]
        The gradient adjustment weight vector.

    Notes
    -----
    See Adam White's PhD thesis, pg. 170-174 for a definition and discussion.
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
        self.z = np.zeros(self.n)
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

        Returns
        -------
        delta : float
            The temporal difference error from the update.
        
        Notes
        -----
        Features (`x` and `xp`) are assumed to be 1D arrays of length `self.n`.
        Other parameters are floats but are generally expected to be in the 
        interval [0, 1].
        """
        delta = r + gm_p*np.dot(self.theta, xp) - np.dot(self.theta, x)
        self.e = rho*(lm*gm*self.e + x)
        self.z = lm*gm*self.z + x 
        self.w += alpha*(delta*self.e + (gm_p*xp - x)*np.dot(self.z - self.e), self.h)
        self.h += beta*(delta*self.e + (gm_p*xp - x)*np.dot(self.z, self.h)

        return delta

    def reset(self):
        """Reset weights, traces, and other parameters."""
        self.e = np.zeros(self.n)
        self.z = np.zeros(self.n)
        self.w = np.zeros(self.n)
        self.h = np.zeros(self.n)

