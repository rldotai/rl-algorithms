/**
 * TOE-TD(lambda): True-online emphatic TD(lambda), an off-policy learning algorithm.
 * See external documentation in TOETD.pdf on the web.
 * @author Rich Sutton, September 2014.
 * Compile with gcc TOETD.cpp -c
 */

class TOETD
{
  //instance variables:
  double *theta;              // main weight vector
  double *e;                  // eligibility trace vector
  int n;                      // dimensionality of the vectors
  double F;                   // scalar memory for the emphasis algorithm
  double D, gamma;            // auxiliary saved scalars from one step to the next

public:

  TOETD(int nArg, double I) {
    n = nArg;
    e = new double[n];
    theta = new double[n];
    for (int i=0; i<n; i++) theta[i]=e[i]=0;
    F = D = gamma = 0;
  }

  void learn(double alpha, double I, double lambda, double phi[], double rho, double R, double phiPrime[], double gammaPrime)
  {
    double Delta_i; // here a scalar, to avoid allocating an extra vector
    double delta = R + gammaPrime*dot(theta,phiPrime) - dot(theta,phi);
    F = F + I;
    double M = lambda*I + (1-lambda)*F;
    double S = rho*alpha*M * (1 - rho*gamma*lambda*dot(phi,e));
    double newD = 0;
    for (int i=0; i<n; i++) {
      e[i] = rho*gamma*lambda*e[i] + S*phi[i];
      Delta_i = delta*e[i] + D * (e[i] - rho*alpha*M*phi[i]);
      theta[i] += Delta_i;
      newD += Delta_i*phiPrime[i];
    }
    D = newD;
    F *= rho*gammaPrime;
    gamma = gammaPrime;
  }

  double predict(double phi[]) {
    return dot(theta,phi);
  }

  double dot(double v1[], double v2[]) {
    // inner product of two vectors of n components
    double sum = 0;
    for (int i=0; i<n; i++)
      sum += v1[i]*v2[i];
    return sum;
  }

  ~TOETD() {
    delete [] theta;
    delete [] e;
  }

};

