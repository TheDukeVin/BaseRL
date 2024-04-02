
#include "environment.h"

void Environment::randomizeOptions(){
    int allTypes[numType];
    for(int i=0; i<numType; i++){
        allTypes[i] = i;
    }
    shuffle(allTypes, allTypes + numType, default_random_engine{dev()});
    for(int i=0; i<numOption; i++){
        options[i] = allTypes[i];
    }
}

Environment::Environment(){
    timeIndex = 0;
    endState = false;

    for(int i=0; i<numType; i++){
        gems[i] = 0;
    }
    randomizeOptions();
}

string Environment::toString(){
    string s = "Gems: ";
    for(int i=0; i<numType; i++){
        s += to_string(gems[i]) + ' ';
    }
    s += "Options: ";
    for(int i=0; i<numOption; i++){
        s += to_string(options[i]) + ' ';
    }
    return s;
}

vector<int> Environment::validActions(){
    vector<int> actions;
    for(int i=0; i<numOption; i++){
        actions.push_back(i);
    }
    return actions;
}

double Environment::makeAction(int action){
    double reward = 1 + 2*gems[options[action]] - price[action];
    gems[options[action]] ++;
    randomizeOptions();

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
    for(int i=0; i<numOption; i++){
        features[i*numType + options[i]] = 1;
    }
    for(int i=0; i<numType; i++){
        features[numOption*numType + i*timeHorizon + gems[i]] = 1;
    }
}