
#include "environment.h"

Environment::Environment(){
    timeIndex = 0;
    endState = false;

    actionState = MOVE_STATE;
    currAgent = 0;
    arrowAgent = -1;

    for(int i=0; i<boardHeight; i++){
        for(int j=0; j<boardWidth; j++){
            grid[i][j] = -1;
        }
    }

    amazon[0][0] = Pos(0, 2);
    amazon[0][1] = Pos(3, 0);
    amazon[1][0] = Pos(5, 3);
    amazon[1][1] = Pos(2, 5);

    for(auto ag : {0,1}){
        for(auto amz : {0,1}){
            grid[amazon[ag][amz].x][amazon[ag][amz].y] = ag;
        }
    }
}

string Environment::toString(){
    string s = "";
    for(int i=0; i<boardHeight; i++){
        for(int j=0; j<boardWidth; j++){
            if(grid[i][j] == EMPTY){
                s += ". ";
            }
            else if(grid[i][j] == FILLED){
                s += "X ";
            }
            else{
                s += to_string(grid[i][j]) + ' ';
            }
        }
        s += '\n';
    }
    return s;
}

string Environment::toCode(){
    return "";
}

vector<Pos> Environment::reachableCells(Pos start){
    vector<Pos> cells;
    for(auto dr : {-1, 0, 1}){
        for(auto dc : {-1, 0, 1}){
            if(dr == 0 && dc == 0) continue;
            for(int i=1;;i++){
                Pos newPos(start.x + dr*i, start.y + dc*i);
                if(!newPos.inBounds() || grid[newPos.x][newPos.y] != EMPTY) break;
                cells.push_back(newPos);
            }
        }
    }
    return cells;
}

vector<int> Environment::validActions(int agentID){
    if(currAgent != agentID) return vector<int>{0};
    vector<int> actions;
    if(actionState == MOVE_STATE){
        for(int i=0; i<numAmazons; i++){
            for(auto p : reachableCells(amazon[agentID][i])){
                actions.push_back(i*boardSize + p.index());
            }
        }
    }
    else{
        for(auto p : reachableCells(amazon[agentID][arrowAgent])){
            actions.push_back(p.index());
        }
    }
    return actions;
}

void Environment::makeAction(int* action, double* reward){
    for(int i=0; i<numAgents; i++){
        reward[i] = 0;
    }

    if(actionState == MOVE_STATE){
        int posIndex = action[currAgent] % boardSize;
        int amzIndex = action[currAgent] / boardSize;

        grid[amazon[currAgent][amzIndex].x][amazon[currAgent][amzIndex].y] = EMPTY;

        amazon[currAgent][amzIndex] = Pos(posIndex);
        grid[amazon[currAgent][amzIndex].x][amazon[currAgent][amzIndex].y] = currAgent;
        
        actionState = ARROW_STATE;
        arrowAgent = amzIndex;
    }
    else{
        Pos arrow(action[currAgent]);
        grid[arrow.x][arrow.y] = FILLED;
        actionState = MOVE_STATE;
        currAgent = 1-currAgent;
        arrowAgent = -1;
    }

    if(actionState == MOVE_STATE){
        if(validActions(currAgent).size() == 0){
            reward[currAgent] = -1;
            reward[1-currAgent] = 1;
            endState = true;
        }
    }

    timeIndex ++;
    if(timeIndex == timeHorizon){
        endState = true;
    }
}

void Environment::getFeatures(int agentID, double* features){
    assert(numAgents == 2);
    for(int i=0; i<numFeatures; i++){
        features[i] = 0;
    }

    for(int i=0; i<numAgents; i++){
        for(int j=0; j<numAmazons; j++){
            features[(i*numAmazons + j)*boardSize + amazon[i][j].x*boardWidth + amazon[i][j].y] = 1;
        }
    }
    for(int i=0; i<boardHeight; i++){
        for(int j=0; j<boardWidth; j++){
            features[2*numAmazons*boardSize + i*boardWidth + j] = (grid[i][j] == FILLED);
            features[(2*numAmazons+1)*boardSize + i*boardWidth + j] = currAgent;
        }
    }
    if(actionState == ARROW_STATE){
        features[(2*numAmazons + 2)*boardSize + amazon[currAgent][arrowAgent].x*boardWidth + amazon[currAgent][arrowAgent].y] = 1;
    }
}