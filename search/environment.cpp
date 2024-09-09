
#include "environment.h"

void Environment::randomizeToken(){
    while(true){
        token = Pos(randomN(boardx), randomN(boardy));
        if(token != agent){
            break;
        }
    }
}

Environment::Environment(){
    timeIndex = 0;
    endState = false;

    agent = Pos(boardx/2, boardy/2);
    randomizeToken();
}

string Environment::toString(){
    return "Time: " + to_string(timeIndex) + " Agent: " + to_string(agent.x) + " " + to_string(agent.y) + " Token: " + to_string(token.x) + " " + to_string(token.y) + "\n";
}

string Environment::toCode(){
    return to_string(timeIndex) + " " + to_string(agent.x) + " " + to_string(agent.y) + " " + to_string(token.x) + " " + to_string(token.y) + "\n";
}

vector<int> Environment::validActions(){
    vector<int> actions;
    for(int i=0; i<4; i++){
        Pos nbr = agent.shift(i);
        if(nbr.inBounds()){
            actions.push_back(i);
        }
    }
    return actions;
}

double Environment::makeAction(int action){
    double reward = 0;

    agent = agent.shift(action);
    if(agent == token){
        reward = 1;
        randomizeToken();
    }

    timeIndex ++;
    if(timeIndex == timeHorizon){
        endState = true;
    }
    return reward;
}

void Environment::getFeatures(double* features){
    for(int i=0; i<numFeatures; i++){
        features[i] = 0;
    }

    features[agent.index()] = 1;

    if(abs(agent.x - token.x) <= proximity && abs(agent.y - token.y) <= proximity){
        features[boardSize + token.index()] = 1;
    }
}