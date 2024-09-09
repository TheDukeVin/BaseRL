
#include "/Users/kevindu/Desktop/Coding/Tests:experiments/PG_test/common.h"

#ifndef environment_h
#define environment_h

const int numAgents = 0;
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
    string toCode();
    vector<int> validActions(int agentID);
    void makeAction(int* action, double* reward); // alters reward array
    void getFeatures(int agentID, double* features); // alters features array
};

#endif