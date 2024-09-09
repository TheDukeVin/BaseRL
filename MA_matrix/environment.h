
#include "/Users/kevindu/Desktop/Coding/Tests:experiments/PG_test/common.h"

#ifndef environment_h
#define environment_h

const int numAgents = 2;
const int timeHorizon = 1;
const double discountFactor = 1;
const int numActions = 3;
const int numFeatures = 2;

const double payoff0[numActions][numActions] = {
    {0.5, -1,  1},
    {  1,  0, -1},
    { -1,  1,  0}
};

class Environment{
public:
    int timeIndex;
    bool endState;

    Environment();
    string toString();
    string toCode();
    vector<int> validActions(int agentID);
    void makeAction(int* action, double* reward); // alters reward array
    void getFeatures(int agentID, double* features); // alters features array
};

#endif