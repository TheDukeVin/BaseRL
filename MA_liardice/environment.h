
#include "/Users/kevindu/Desktop/Coding/Tests:experiments/PG_test/common.h"

#ifndef environment_h
#define environment_h

const int diceSize = 6;

const int numAgents = 2;
const int timeHorizon = diceSize*2;
const double discountFactor = 1;
const int numActions = diceSize*2+1;
const int numFeatures = diceSize*5 + 2;

const int CALL_ACTION = 2*diceSize;

class Environment{
public:
    int diceVals[numAgents];
    int currAgent;
    int bets[timeHorizon];
    
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