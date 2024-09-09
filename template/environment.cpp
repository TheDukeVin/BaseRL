
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

vector<int> Environment::validActions(){
    return vector<int>();
}

double Environment::makeAction(int action){
    double reward = 0;
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
}