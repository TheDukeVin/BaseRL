
#include "/Users/kevindu/Desktop/Coding/Tests:experiments/PG_test/common.h"

#ifndef environment_h
#define environment_h

// Learn a permutation

const int timeHorizon = 5;
const double discountFactor = 1;
const int numActions = 3;

const int numFeatures = 3;
const int perm[3] = {2, 0, 1};

class Environment{
public:
    int timeIndex;
    bool endState;

    int currInput;

    Environment();
    string toString();
    vector<int> validActions();
    double makeAction(int action); // returns reward
    void getFeatures(double* features);
};

#endif