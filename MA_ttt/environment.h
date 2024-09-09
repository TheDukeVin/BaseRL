
#include "/Users/kevindu/Desktop/Coding/Tests:experiments/PG_test/common.h"

#ifndef environment_h
#define environment_h

const int boardWidth = 3;
const int reqConn = 3;
const int boardSize = boardWidth * boardWidth;

const int numAgents = 2;
const int timeHorizon = boardSize;
const double discountFactor = 1;
const int numActions = boardSize;
const int numFeatures = boardSize*2 + numAgents;

class Environment{
public:
    int board[boardWidth][boardWidth];
    int currAgent;

    bool checkWin(int player);

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