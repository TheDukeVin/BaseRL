
#include "environment.h"

Environment::Environment(){
    timeIndex = 0;
    endState = false;

    call = randomN(size);
    lastCall = -1;
}

string Environment::toString(){
    return "Time: " + to_string(timeIndex) + " Call: " + to_string(call) + "\n";
}

vector<int> Environment::validActions(){
    vector<int> actions;
    for(int i=0; i<size; i++){
        actions.push_back(i);
    }
    return actions;
}

double Environment::makeAction(int action){
    double reward = 0;

    if(action == lastCall){
        reward = 1;
    }
    lastCall = call;
    call = randomN(size);

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

    features[call] = 1;
}