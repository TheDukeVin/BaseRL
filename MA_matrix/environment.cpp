
#include "environment.h"

Environment::Environment(){
    timeIndex = 0;
    endState = false;
}

string Environment::toString(){
    return to_string(timeIndex) + '\n';
}

string Environment::toCode(){
    return "";
}

vector<int> Environment::validActions(int agentID){
    vector<int> actions;
    for(int i=0; i<numActions; i++){
        actions.push_back(i);
    }
    return actions;
}

void Environment::makeAction(int* action, double* reward){
    reward[0] = payoff0[action[0]][action[1]];
    reward[1] = -reward[0];
    timeIndex ++;
    if(timeIndex == timeHorizon){
        endState = true;
    }
}

void Environment::getFeatures(int agentID, double* features){
    for(int i=0; i<numFeatures; i++){
        features[i] = 0;
    }
    features[agentID] = 1;
}