
#include "/Users/kevindu/Desktop/Coding/Tests:experiments/PG_test/common.h"

#ifndef environment_h
#define environment_h

const int boardx = 10;
const int boardy = 10;
const int boardSize = boardx*boardy;
const double radarProb = 0.1;
const double tokenMoveProb = 0.5;

const int timeHorizon = 100;
const double discountFactor = 0.99;
const int numActions = 4;
const int numFeatures = boardSize*2;

const int dir[4][2] = {{0,1}, {1,0}, {0,-1}, {-1,0}};

class Pos{
public:
    int x, y;

    Pos(){
        x = y = -1;
    }

    Pos(int _x, int _y){
        x = _x; y = _y;
    }

    bool inBounds(){
        return 0 <= x && x < boardx && 0 <= y && y < boardy;
    }

    Pos shift(int d){
        return Pos(x + dir[d][0], y + dir[d][1]);
    }

    int index() const{
        return x*boardy + y;
    }

    friend bool operator == (const Pos& p, const Pos& q){
        return (p.x == q.x) && (p.y == q.y);
    }

    friend bool operator != (const Pos& p, const Pos& q){
        return (p.x != q.x) || (p.y != q.y);
    }
};

class Environment{
private:
    Pos token;

public:
    int timeIndex;
    bool endState;

    Pos agent;
    bool radar;

    void randomizeRadar();
    void randomizeToken();

    Environment();
    string toString();
    string toCode();
    vector<int> validActions();
    double makeAction(int action); // returns reward
    void getFeatures(double* features);
};

#endif