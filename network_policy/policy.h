
#include "../common.h"
#include "lstm.h"

#ifndef policy_h
#define policy_h

// network policy

class Policy{
private:
    int numInputs;
    int numOutputs;
    const static double constexpr entropyConstant = 0.01;

    LSTM::Model net;
    LSTM::Data* netInput;
    LSTM::Data* netOutput;
    LSTM::Model structure;

public:

    double* features;
    double* outputPolicy;

    Policy(){}
    Policy(int numIn, int numOut, LSTM::Model structure_);
    
    void resetGrad();
    void evaluate(vector<int> validActions);
    void accGrad(vector<int> validActions, int action, double value);
    void update(double scale, double momentum);
};

#endif