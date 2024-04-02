
#include "/Users/kevindu/Desktop/Coding/Tests:experiments/PG_test/common.h"

#ifndef environment_h
#define environment_h

const int timeHorizon = 0;
const double discountFactor = 0;
const int numActions = 0;
const int numFeatures = 0;

class Environment{
public:
    int timeIndex;
    bool endState;

    Environment();
    string toString();
    vector<int> validActions();
    double makeAction(int action); // returns reward
    void getFeatures(double* features);
};

#endif