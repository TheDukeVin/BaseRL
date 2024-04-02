
#include "environment.h"

Environment::Environment(){
    timeIndex = 0;
    endState = false;
    currInput = randomN(3);
}

string Environment::toString(){
    return "Time: " + to_string(timeIndex) + " Input: " + to_string(currInput);
}

vector<int> Environment::validActions(){
    return vector<int>{0, 1, 2};
}

double Environment::makeAction(int action){
    double reward = 0;
    if(action == perm[currInput]){
        reward = 1;
    }
    currInput = randomN(3);
    return reward;
}

void Environment::getFeatures(double* features){
    for(int i=0; i<3; i++){
        features[i] = 0;
    }
    features[currInput] = 1;
}