
#include "environment.h"

Environment::Environment(){
    timeIndex = 0;
    endState = false;
}

string Environment::toString(){
    return "";
}

string Environment::toCode(){
    return "";
}

vector<int> Environment::validActions(int agentID){
    return vector<int>();
}

void Environment::makeAction(int* action, double* reward){
    for(int i=0; i<numAgents; i++){
        reward[i] = 0;
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
}