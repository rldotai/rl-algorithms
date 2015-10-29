/*
* ETD(lambda): Emphatic Temporal Difference Learning
*
* @author Brendan Bennett, Rich Sutton, October 2015.
*
* CHANGES FROM TOETD.cpp
*   - renamed some variables
*   - removed `gamma` as object variable, since it was unused
*   - rearranged parameters in `learn()` so that `phi`, `r`, `phi_p` come first
*/

class ETD
{
    int n;
    double *theta;
    double *e;
    double F;
    double D;

public:
    ETD(int fvec_length) {
        n = fvec_length;
        e = new double[n];
        theta = new double[n];

        // initialize weight vector and traces
        for (int i=0; i<n; i++) {
            e[i] = 0;
            theta[i] = 0;
        }
        // initialize scalar variables
        F = 0;
        D = 0;
    }

    void learn(double phi[], double r, double phi_p[],
               double alpha, double gamma, double gamma_p, double I,
               double lambda, double rho) {
        // perform learning update

        F = F + I; // avoid keeping track of previous timestep's rho
        double delta = r + gamma_p * dot(theta, phi_p) - dot(theta, phi);
        double M = lambda*I + (1-lambda)*F;
        double S = rho*alpha*M*(1 - rho*gamma*lambda*dot(phi, e));
        double D_p = 0;

        // update weights and traces
        double delta_i;
        for (int i=0; i<n; i++) {
            e[i] = rho*gamma*lambda*e[i] + S*phi[i];
            delta_i = delta*e[i] + D * (e[i] - rho*alpha*M*phi[i]);
            theta[i] += delta_i;
            D_p += delta_i * phi_p[i];
        }
        // prepare for next iteration
        D = D_p;
        F *= rho*gamma_p;
    }

    double predict(double fvec[]) {
        // return the prediction for a feature vector
        return dot(theta, fvec);
    }

    double dot(double v1[], double v2[]) {
        // inner product of two vectors of `n` components
        double ret = 0;
        for (int i=0; i<n; i++) {
            ret += v1[i]*v2[i];
        }
        return ret;
    }

    ~ETD() {
        delete [] e;
        delete [] theta;
    }
}