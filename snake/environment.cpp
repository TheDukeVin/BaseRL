
#include "environment.h"

void Environment::setGridValue(Pos p, int val){
    grid[p.x][p.y] = val;
}

int Environment::getGridValue(Pos p){
    return grid[p.x][p.y];
}

void Environment::randomizeApple(){
    if(snake.size == boardSize){
        return;
    }
    while(true){
        apple = Pos(randomN(boardx), randomN(boardy));
        if(getGridValue(apple) == -1){
            break;
        }
    }
}

Environment::Environment(){
    timeIndex = 0;
    endState = false;

    snake.size = 2;
    snake.head = Pos(boardx/2, 1);
    snake.tail = Pos(boardx/2, 0);
    
    for(int i=0; i<boardx; i++){
        for(int j=0; j<boardy; j++){
            grid[i][j] = -1;
        }
    }

    setGridValue(snake.head, 4);
    setGridValue(snake.tail, 0);

    randomizeApple();
    applySym();
}

string Environment::toString(){
    string s = "";
    s += "Timer: " + to_string(timeIndex) + '\n';
    s += "Snake size: " + to_string(snake.size) + '\n';
    char output[2*boardx+1][2*boardy+1];
    for(int i=0; i<2*boardx+1; i++){
        for(int j=0; j<2*boardy+1; j++){
            output[i][j] = ' ';
        }
    }
    for(int i=0; i<2*boardx+1; i++){
        output[i][0] = '#';
        output[i][2*boardy] = '#';
    }
    for(int j=0; j<2*boardy+1; j++){
        output[0][j] = '#';
        output[2*boardx][j] = '#';
    }
    char body = 'x';
    char head = 'X';
    for(int i=0; i<boardx; i++){
        for(int j=0; j<boardy; j++){
            int val = getGridValue(Pos(i, j));
            char out;
            if(apple == Pos(i, j)){
                out = 'A';
            }
            else if(val == -1){
                out = '.';
            }
            else if(val == 4){
                out = head;
            }
            else{
                out = body;
                char bar;
                if(val%2 == 0) bar = '-';
                else bar = '|';
                output[2*i+1 + dir[val][0]][2*j+1 + dir[val][1]] = bar;
            }
            output[2*i+1][2*j+1] = out;
        }
    }
    for(int i=0; i<2*boardx+1; i++){
        for(int j=0; j<2*boardy+1; j++){
            s += output[i][j];
        }
        s += '\n';
    }
    s += '\n';
    return s;
}

vector<int> Environment::validActions(){
    vector<int> actions;
    for(int i=0; i<4; i++){
        Pos nbr = snake.head.shift(i);
        if(nbr.inBounds() && (getGridValue(nbr) == -1)){
            actions.push_back(i);
        }
    }
    return actions;
}

double Environment::makeAction(int action){
    double reward = 0;

    Pos newHead = snake.head.shift(action);
    setGridValue(snake.head, action);
    snake.head = newHead;
    setGridValue(newHead, 4);

    
    if(newHead == apple){
        snake.size ++;
        reward = 1;
        randomizeApple();
    }
    else{
        int tailDir = getGridValue(snake.tail);
        setGridValue(snake.tail, -1);
        snake.tail = snake.tail.shift(tailDir);
    }

    if(validActions().size() == 0){
        endState = true;
        if(snake.size == boardSize){
            reward = outcomeReward;
        }
        else{
            reward = -outcomeReward;
        }
    }

    timeIndex ++;
    if(timeIndex == timeHorizon){
        endState = true;
    }

    applySym();

    return reward;
}

void Environment::getFeatures(double* features){
    for(int i=0; i<numFeatures; i++){
        features[i] = 0;
    }

    for(int i=0; i<boardx; i++){
        for(int j=0; j<boardy; j++){
            if(grid[i][j] != -1){
                features[grid[i][j]*boardSize + i*boardy + j] = 1;
            }
        }
    }

    features[4*boardSize + snake.head.index()] = 1;
    features[5*boardSize + snake.tail.index()] = 1;
    features[6*boardSize + apple.index()] = 1;

    for(int i=0; i<boardx; i++){
        for(int j=0; j<boardy; j++){
            features[7*boardSize + i*boardy + j] = pow(discountFactor, timeHorizon - timeIndex);
        }
    }
}



int randomSym(){
    assert(boardx == boardy);
    return randomN(8);
}

int symAction(int action, int symID){
    return (symDir[symID][0]*action + symDir[symID][1] + 4) % 4;
}

Pos transform(Pos p, int symID){
    int x = sym[symID][0][0]*p.x + sym[symID][0][1]*p.y + sym[symID][0][2];
    int y = sym[symID][1][0]*p.x + sym[symID][1][1]*p.y + sym[symID][1][2];
    return Pos(x, y);
}

void Environment::applySym(){
    int symID = randomSym();
    int newSnake[boardx][boardy];
    for(int i=0; i<boardx; i++){
        for(int j=0; j<boardy; j++){
            Pos newPos = transform(Pos(i, j), symID);
            int new_dir;
            if(0 <= grid[i][j] && grid[i][j] < 4){
                new_dir = symAction(grid[i][j], symID);
            }
            else{
                new_dir = grid[i][j];
            }
            newSnake[newPos.x][newPos.y] = new_dir;
        }
    }
    for(int i=0; i<boardx; i++){
        for(int j=0; j<boardy; j++){
            grid[i][j] = newSnake[i][j];
        }
    }

    snake.head = transform(snake.head, symID);
    snake.tail = transform(snake.tail, symID);
    apple = transform(apple, symID);
}