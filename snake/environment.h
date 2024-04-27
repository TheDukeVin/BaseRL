
#include "../common.h"

#ifndef environment_h
#define environment_h

const int boardx = 10;
const int boardy = 10;
const int boardSize = boardx*boardy;
const double outcomeReward = 5;

const int timeHorizon = 1200;
const double discountFactor = 0.99;
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

class PosHash{
public:
    size_t operator()(const Pos& p) const {
        return p.index();
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

const int m = boardx-1;
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
    void getFeatures(double* features, int symID=0);

    double potential();
};

class SnakeDFS{
public:
    Environment env;
    unordered_set<Pos, PosHash> visited;

    SnakeDFS(Environment env_);
    void DFS(Pos p);
};

#endif