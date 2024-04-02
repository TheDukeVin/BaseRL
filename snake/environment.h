
#include "../common.h"

#ifndef environment_h
#define environment_h

const int boardx = 6;
const int boardy = 6;
const int boardSize = boardx*boardy;
const double outcomeReward = 5;

const int timeHorizon = 200;
const double discountFactor = 0.98;
const int numActions = 4;
const int numFeatures = boardSize*7;

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

    int index(){
        return x*boardy + y;
    }

    friend bool operator == (const Pos& p, const Pos& q){
        return (p.x == q.x) && (p.y == q.y);
    }

    friend bool operator != (const Pos& p, const Pos& q){
        return (p.x != q.x) || (p.y != q.y);
    }
};

class Snake{
public:
    int size;
    Pos head;
    Pos tail;

    Snake(){
        size = -1;
    }
};

class Environment{
public:
    int timeIndex;
    bool endState;
    
    Snake snake;
    Pos apple;

    // -1 = not snake. 0 to 3 = snake unit pointing to next unit. 4 = head.
    int grid[boardx][boardy];

    void setGridValue(Pos p, int val);
    int getGridValue(Pos p);
    void randomizeApple();

    Environment();
    string toString();
    vector<int> validActions();
    double makeAction(int action); // returns reward
    void getFeatures(double* features);
};

#endif