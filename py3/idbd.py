"""
Incremental Delta-Bar-Delta (IDBD), a stepsize adjustment algorithm.
Not strictly a reinforcement learning algorithm, since it assumes more of an 
online learning setting (in contrast to TD(λ), which attempts to solve the 
Bellman equation, which depends on the current state as well as the next state
in sequence: δ(t) = r(t) + v(t) - γv(t+1), vs. δ(t) = y(t) - x(t) * w(t)).

Taken from:
"Adapting Bias by Gradient Descent: An Incremental Version of Delta-Bar-Delta", 
Richard Sutton
Proceedings of Tenth National Conf. on Artificial Intelligence, pp. 171–176, 
MIT Press, 1992.
"""

class IDBD:
    """
    Incremental Delta-Bar-Delta, or IDBD.

    Attributes
    ----------
    n : int 

    alpha : Vector[float]
        The vector of per-weight stepsizes.
    beta : Vector[float]
        The vector of logarithmic per-weight stepsizes.
    w : Vector[float]
        Weight vector.
    h : Vector[float]
        Update memory trace.
    eta : float 
        Meta stepsize parameter.
    """
    def __init__(self, n, eta=1):
        self.n = n 
        self.eta = eta
        self.reset()

    def reset(self):
        # What should beta be initialized to? Should `w` be zeros or random?
        self.beta = (-1/self.n)*np.ones(self.n)
        self.h = np.zeros(self.n)
        self.w = np.zeros(self.n)

    def update(self, x, delta):
        self.beta += self.eta * self.h * delta * x
        self.alpha = np.exp(self.beta)
        self.w += self.alpha * delta * x 
        self.h = self.h * np.max(0, 1 - self.alpha * x**2) + self.alpha*delta*x