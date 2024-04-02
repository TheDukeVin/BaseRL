
#include "/Users/kevindu/Desktop/Coding/Tests:experiments/PG_test/common.h"

#ifndef environment_h
#define environment_h

const int numType = 5;
const int numOption = 3;
const int price[numOption] = {0, 1, 2};

const int timeHorizon = 10;
const double discountFactor = 1;
const int numActions = numOption;
const int numFeatures = numOption*numType + timeHorizon*numType;

class Environment{
public:
    int timeIndex;
    bool endState;

    int gems[numType];
    int options[numOption];

    void randomizeOptions();

    Environment();
    string toString();
    vector<int> validActions();
    double makeAction(int action); // returns reward
    void getFeatures(double* features);
};

#endif