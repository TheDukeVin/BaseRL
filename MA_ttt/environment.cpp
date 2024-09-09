
#include "environment.h"

Environment::Environment(){
    timeIndex = 0;
    endState = false;
    currAgent = 0;

    for(int i=0; i<boardWidth; i++){
        for(int j=0; j<boardWidth; j++){
            board[i][j] = -1;
        }
    }
}

string Environment::toString(){
    string s = "";
    for(int i=0; i<boardWidth; i++){
        for(int j=0; j<boardWidth; j++){
            if(board[i][j] == -1) s += ".";
            if(board[i][j] == 0) s += "X";
            if(board[i][j] == 1) s += "O";
        }
        s += "\n";
    }
    return s;
}

string Environment::toCode(){
    return "";
}

vector<int> Environment::validActions(int agentID){
    if(agentID != currAgent){
        return vector<int>{0};
    }
    vector<int> moves;
    for(int i=0; i<boardWidth; i++){
        for(int j=0; j<boardWidth; j++){
            if(board[i][j] == -1){
                moves.push_back(i * boardWidth + j);
            }
        }
    }
    return moves;
}

void Environment::makeAction(int* action, double* reward){
    for(int i=0; i<numAgents; i++){
        reward[i] = 0;
    }
    board[action[currAgent] / boardWidth][action[currAgent] % boardWidth] = currAgent;
    if(checkWin(currAgent)){
        reward[currAgent] = 1;
        reward[1-currAgent] = -1;
        endState = true;
    }
    else{
        currAgent = 1-currAgent;
    }

    timeIndex ++;
    if(timeIndex == timeHorizon){
        endState = true;
    }
}

void Environment::getFeatures(int agentID, double* features){
    for(int i=0; i<numFeatures; i++){
        features[i] = 0;
    }
    for(int i=0; i<boardWidth; i++){
        for(int j=0; j<boardWidth; j++){
            if(board[i][j] == 0){
                features[i*boardWidth + j] = 1;
            }
            if(board[i][j] == 1){
                features[boardSize + i*boardWidth + j] = 1;
            }
        }
    }
    features[2*boardSize + agentID] = 1;
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
                shiftGrid[i][j] = -1;
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