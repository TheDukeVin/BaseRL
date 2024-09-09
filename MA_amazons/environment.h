
#include "../common.h"

#ifndef environment_h
#define environment_h

const int boardHeight = 6;
const int boardWidth = 6;
const int boardSize = boardHeight * boardWidth;
const int numAmazons = 2;

const int numAgents = 2;
const int timeHorizon = boardSize*2;
const double discountFactor = 1;
const int numActions = boardSize*numAmazons;
const int numFeatures = boardSize*(2*numAmazons + 3);

class Pos{
public:
    int x, y;

    Pos(){
        x = y = -1;
    }

    Pos(int index){
        x = index / boardWidth;
        y = index % boardWidth;
    }

    Pos(int _x, int _y){
        x = _x; y = _y;
    }

    bool inBounds(){
        return 0 <= x && x < boardHeight && 0 <= y && y < boardWidth;
    }

    int index() const{
        return x*boardWidth + y;
    }

    friend bool operator == (const Pos& p, const Pos& q){
        return (p.x == q.x) && (p.y == q.y);
    }

    friend bool operator != (const Pos& p, const Pos& q){
        return (p.x != q.x) || (p.y != q.y);
    }
};

const int EMPTY = -1;
const int FILLED = -2;

const int MOVE_STATE = 0;
const int ARROW_STATE = 1;

class Environment{
public:
    int grid[boardHeight][boardWidth];
    Pos amazon[numAgents][numAmazons];
    int actionState;
    int currAgent;
    int arrowAgent;

    vector<Pos> reachableCells(Pos start);

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