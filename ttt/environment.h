
#include "../common.h"

#ifndef environment_h
#define environment_h

const int boardWidth = 3;
const int reqConn = 3;
const int boardSize = boardWidth * boardWidth;

const int timeHorizon = boardSize;
const double discountFactor = 1;
const int numActions = boardSize;
const int numFeatures = boardSize*2;

class Environment{
public:
    int board[boardWidth][boardWidth];

    void randomOpponentMove();
    bool checkWin(int player);

    int timeIndex;
    bool endState;

    Environment();
    string toString();
    string toCode(){return "";}
    vector<int> validActions();
    double makeAction(int action); // returns reward
    void getFeatures(double* features);
};

#endif