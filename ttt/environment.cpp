
#include "environment.h"

void Environment::randomOpponentMove(){
    vector<int> moves = validActions();
    assert(moves.size() > 0);
    int index = randomN(moves.size());
    int move = moves[index];
    board[move / boardSize][move % boardSize] = -1;
}

Environment::Environment(){
    timeIndex = 0;
    endState = false;

    for(int i=0; i<boardWidth; i++){
        for(int j=0; j<boardWidth; j++){
            board[i][j] = 0;
        }
    }
    if(randomN(2) == 0) randomOpponentMove();
}

string Environment::toString(){
    string s = "";
    for(int i=0; i<boardWidth; i++){
        for(int j=0; j<boardWidth; j++){
            if(board[i][j] == 1) s += "X";
            if(board[i][j] == 0) s += ".";
            if(board[i][j] == -1) s += "O";
        }
        s += "\n";
    }
    return s;
}

vector<int> Environment::validActions(){
    vector<int> moves;
    for(int i=0; i<boardWidth; i++){
        for(int j=0; j<boardWidth; j++){
            if(board[i][j] == 0){
                moves.push_back(i * boardWidth + j);
            }
        }
    }
    return moves;
}

bool Environment::checkWin(int player){
    // Transform the grid
    int shiftGrid[boardWidth][boardWidth*3];
    int transform[4][4] = {
        {1, 0, 0, 1},
        {0, 1, 1, 0},
        {1, 0, 1, 1},
        {1, 0, -1, 1}
    };
    for(int trans=0; trans<4; trans++){
        for(int i=0; i<boardWidth; i++){
            for(int j=0; j<boardWidth*3; j++){
                shiftGrid[i][j] = 0;
            }
        }
        for(int i=0; i<boardWidth; i++){
            for(int j=0; j<boardWidth; j++){
                shiftGrid[i*transform[trans][0] + j*transform[trans][1]][boardWidth + i*transform[trans][2] + j*transform[trans][3]] = board[i][j];
            }
        }
        for(int i=0; i<boardWidth*3; i++){
            int conn = 0;
            for(int j=0; j<boardWidth; j++){
                if(shiftGrid[j][i] == player){
                    conn ++;
                    if(conn == reqConn) return true;
                }
                else{
                    conn = 0;
                }
            }
        }
    }
    return false;
}

double Environment::makeAction(int action){
    double reward = 0;
    board[action / boardSize][action % boardSize] = 1;
    if(checkWin(1)){
        reward = 1;
        endState = true;
    }
    else if(validActions().size() > 0){
        randomOpponentMove();
        if(checkWin(-1)){
            reward = -1;
            endState = true;
        }
    }
    timeIndex ++;
    if(validActions().size() == 0){
        endState = true;
    }
    return reward;
}

void Environment::getFeatures(double* features){
    for(int i=0; i<numFeatures; i++){
        features[i] = 0;
    }
    for(int i=0; i<boardWidth; i++){
        for(int j=0; j<boardWidth; j++){
            if(board[i][j] == 1){
                features[i*boardWidth + j] = 1;
            }
            if(board[i][j] == -1){
                features[boardSize + i*boardWidth + j] = 1;
            }
        }
    }
}