
#include "environment.h"

void Environment::randomizeRadar(){
    radar = randUniform() < radarProb;
}

void Environment::randomizeToken(){
    token = Pos(randomN(boardx), randomN(boardy));
}

Environment::Environment(){
    timeIndex = 0;
    endState = false;

    randomizeRadar();
    randomizeToken();

    agent = Pos(boardx/2, boardy/2);
}

string Environment::toString(){
    return "";
}

string Environment::toCode(){
    return to_string(timeIndex) + " " + to_string(agent.x) + " " + to_string(agent.y) + " " + to_string(token.x) + " " + to_string(token.y) + " " + to_string(radar) + "\n";
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

    if(token == agent){
        reward = 1;
        randomizeToken();
    }
    else{
        agent = agent.shift(action);

        vector<int> validDir;
        for(int i=0; i<4; i++){
            Pos nbr = token.shift(i);
            if(nbr.inBounds()){
                validDir.push_back(i);
            }
        }
        if(randUniform() < tokenMoveProb){
            token = token.shift(validDir[randomN(validDir.size())]);
        }
    }
    randomizeRadar();
    

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
    if(radar){
        features[boardSize + token.index()] = 1;
    }
}