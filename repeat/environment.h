
#include "/Users/kevindu/Desktop/Coding/Tests:experiments/PG_test/common.h"

#ifndef environment_h
#define environment_h

const int size = 4;

const int timeHorizon = 5;
const double discountFactor = 1;
const int numActions = size;
const int numFeatures = size;

class Environment{
private:
    int lastCall;

public:
    int timeIndex;
    bool endState;

    int call;

    Environment();
    string toString();
    vector<int> validActions();
    double makeAction(int action); // returns reward
    void getFeatures(double* features);
};

#endif