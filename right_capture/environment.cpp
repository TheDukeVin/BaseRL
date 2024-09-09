
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
        // place token close to agent if captured from the right
        reward = 1;
        int close = 1;
        if(action == 2 && randUniform() < 0.8){
            while(true){
                token = Pos(agent.x + randomN(2*close+1)-close, agent.y + randomN(2*close+1)-close);
                if(token.inBounds() && token != agent) break;
            }
        }
        else{
            randomizeToken();
        }
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
    features[boardSize + token.index()] = 1;
}