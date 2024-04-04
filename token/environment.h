
#include "/Users/kevindu/Desktop/Coding/Tests:experiments/PG_test/common.h"

#ifndef environment_h
#define environment_h

const int size = 10;

const int timeHorizon = 100;
const double discountFactor = 0.9;
const int numActions = 4;
const int numFeatures = 4;

const int dir[4][2] = {{0,1}, {1,0}, {0,-1}, {-1,0}};

class Pos{
public:
    int x, y;

    Pos(){
        x = y = -1;
    }

    Pos(int x_, int y_){
        x = x_; y = y_;
    }

    bool inBounds(){
        return 0 <= x && x < size && 0 <= y && y < size;
    }

    Pos shift(int d){
        return Pos(x + dir[d][0], y + dir[d][1]);
    }

    friend bool operator == (const Pos& p, const Pos& q){
        return (p.x == q.x) && (p.y == q.y);
    }

    friend bool operator != (const Pos& p, const Pos& q){
        return (p.x != q.x) || (p.y != q.y);
    }
};

const int symDir[8][2] = {
    { 1,0},
    { 1,3},
    { 1,2},
    { 1,1},
    {-1,1},
    {-1,2},
    {-1,3},
    {-1,0}
};

const int m = size-1;
const int sym[8][2][3] = {
    {{ 1, 0, 0},{ 0, 1, 0}},
    {{ 0,-1, m},{ 1, 0, 0}},
    {{-1, 0, m},{ 0,-1, m}},
    {{ 0, 1, 0},{-1, 0, m}},
    {{ 0, 1, 0},{ 1, 0, 0}},
    {{ 1, 0, 0},{ 0,-1, m}},
    {{ 0,-1, m},{-1, 0, m}},
    {{-1, 0, m},{ 0, 1, 0}}
};

int randomSym();
Pos transform(Pos p, int symID);
int symAction(int action, int symID); // transforms action in original environment into equivalent action in mirror environment.

class Environment{
public:
    int timeIndex;
    bool endState;

    Pos agent;
    Pos token;

    void randomizeToken();

    Environment();
    string toString();
    vector<int> validActions();
    double makeAction(int action); // returns reward
    void getFeatures(double* features);
};

#endif