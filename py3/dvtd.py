"""
Direct variance TD(λ), also known as TD-δ^2, an online temporal difference
algorithm for estimating the return.

From the paper "Directly Estimating the Variance of the λ-return Using
Temporal-Difference Methods" (available on ArXiV[0]).


Summary
-------

When an agent learns the value function, it is learning the expected value of
the return.
However, learning the variance of the return may also be worthwhile, but until
relatively recently there were not a lot of research in this area.

This algorithm estimates the variance of the λ-return online using two
TD-learners, one which learns the value function, and a second that uses the
squared TD-errors from the first to generate a new approximation target.

When true value function is available, the discounted sum of the squared
TD-errors as a is equivalent to the variance of the return.
If the true value function is not available (e.g., because it is not
representable because of limitations on the function approximation being used)
it may not target the true variance, but often close enough to be useful.


Update Equations
----------------

Here we provide the update equations in pseudo-LaTeX, which turns out to have
been a bit of a poor decision.
If this is difficult to read, either refer to the paper[0] or the code below.

For a feature function `x(.)` that maps states to real-valued vectors, the
estimated value of state `s` is given by `v(s)` expressed as `v(s) = w^T x(s)`.
In a similar fashion we denote the estimated variance of the λ-return by `u(s)`,
with `u(s) = ŵ^T x(s)`.

The update equations for the value estimator are just those of
TD(λ) with accumulating traces:

    δ_{t}   = R_{t+1} + γ_{t+1} w_{t}^T x_{t+1} - w_{t}^{T} x_{t}
    e_{t}   = ρ_{t} (λ_{t} γ_{t} e_{t-1} + x_{t})
    w_{t+1} = w_{t} + α δ_{t} e_{t}

Where:
    - δ refers to the temporal difference error
    - γ is the discount parameter
    - λ is the bootstrapping parameter
    - α is the stepsize parameter
    - w is the weight vector
    - e is the eligibility trace
    - x and r are feature vectors and rewards respectively

The variance estimator is also based on TD(λ), but uses the TD-errors (δ) from
the value estimator as rewards, and has a discount factor that is dependent on
the γ and λ used by the value learner.
We abuse unicode's circumflex accent in order to make the similarity with the
value update equations maximally apparent.

    R̂_{t+1}   = δ_{t}^{2}
    ŷ_{t+1}   = (γ_{t+1} λ_{t+1})^{2}
    ε_{t}     = R̂_{t+1} + ŷ_{t+1} ŵ_{t}^{T} x_{t+1} - ŵ_{t}^{T} x_{t}
    ŵ_{t+1}   = α ε_{t} x_{t}

Where:

    - R̂ is the "reward"
    - ŷ is the discount factor
    - ε is the temporal difference error
    - ŵ is the weight vector
    - Other variables are the same as in the value update equations.


Notes
-----

Here we learn the variance of the λ-return using what is effectively TD(0).
It is simpler to present this way, but more elaborate variations are possible.

For example, you it is possible to use different bootstrapping for each
algorithm, or even learn the variance for one λ-return using the value estimated
with a different value of λ.
This is likely irrelevant to the end-users of this code, but I would feel remiss
if I failed to make a note of it.


References
----------

0. https://arxiv.org/abs/1801.08287
"""
import numpy as np


class DVTD:
    """Direct-Variance Temporal Difference Learning or DVTD(λ).

    Attributes
    ----------
    n : int
        The number of features (and therefore the length of the weight vector).
    z_val : Vector[float]
        The eligibility trace vector for the value estimator.
    w_val : Vector[float]
        The weight vector for the value estimator.
    w_var : Vector[float]
        The weight vector for the variance estimator.

    Notes
    -----
    This version is somewhat simplified for pedagogical reasons; see the paper
    referenced in the file's documentation for the full version.

    The version implemented here uses general value functions (GVFs), meaning that
    the discount factor, γ, and the bootstrapping factor, λ, may be functions
    of state.
    If that seems excessive for your needs, just use constant values for γ and
    λ.

    """
    def __init__(self, n):
        """Initialize the learning algorithm.

        Parameters
        -----------
        n : int
            The number of features, i.e. expected length of the feature vector.
        """
        self.n = n
        self.w_val = np.zeros(self.n)
        self.z_val = np.zeros(self.n)
        self.w_var = np.zeros(self.n)

    def get_value(self, x):
        """Get the approximate value for feature vector `x`."""
        return np.dot(self.w_val, x)

    def get_variance(self, x):
        """Get the approximate variance for feature vector `x`."""
        return np.dot(self.w_var, x)

    def update(self, x, r, xp, alpha, gm, gm_p, lm, lm_p):
        """Update from new experience, i.e. from a transition `(x,r,xp)`.

        Parameters
        ----------
        x : Vector[float]
            The observation/features from the current timestep.
        r : float
            The reward from the transition.
        xp : Vector[float]
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
        lm_p: float
            Lambda prime, abbreviated `lm_p`, is the bootstrapping parameter
            for the next timestep.

        Returns
        -------
        delta : float
            The temporal difference error from the value update.
        delta_var : float
            The temporal difference error from the variance update.

        Notes
        -----
        Features (`x` and `xp`) are assumed to be 1D arrays of length `self.n`.
        Other parameters are floats but are generally expected to be in the
        interval [0, 1].
        """
        delta = r + gm_p*np.dot(self.w, xp) - np.dot(self.w, x)
        self.z = x + gm*lm*self.z
        self.w += alpha*delta*self.z

        r_var = delta**2
        γ_var = (gm_p*lm_p)**2
        delta_var = r_var + γ_var*np.dot(self.w_var, xp) - np.dot(self.w_var, x)
        self.w_var += alpha*delta_var*x
        return delta, delta_var

    def reset(self):
        """Reset weights, traces, and other parameters."""
        self.z_val = np.zeros(self.n)
        self.w_val = np.zeros(self.n)
        self.w_var = np.zeros(self.n)
