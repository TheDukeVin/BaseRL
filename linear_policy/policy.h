
#include "/Users/kevindu/Desktop/Coding/Tests:experiments/PG_test/common.h"

#ifndef policy_h
#define policy_h

// linear policy

class Policy{
public:
    int numInputs;
    int numOutputs;
    const static double constexpr entropyConstant = 0.01;

    double* weights;
    double* weightGrad;


    double* features;
    double* outputPolicy;

    Policy(){}
    Policy(int numIn, int numOut);
    
    void resetGrad();
    void evaluate(vector<int> validActions);
    void accGrad(vector<int> validActions, int action, double value);
    void update(double scale);
};

#endif