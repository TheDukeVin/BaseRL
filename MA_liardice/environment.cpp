
#include "environment.h"

Environment::Environment(){
    timeIndex = 0;
    endState = false;

    diceVals[0] = randomN(diceSize);
    diceVals[1] = randomN(diceSize);

    currAgent = 0;

    for(int i=0; i<timeHorizon; i++){
        bets[i] = -1;
    }
}

string Environment::toString(){
    string s = "Dice: " + to_string(diceVals[0]) + ' ' + to_string(diceVals[1]) + '\n' + "Bets: ";
    for(int i=0; i<timeIndex; i++){
        s += to_string(bets[i]) + ' ';
    }
    return s + '\n';
}

string Environment::toCode(){
    return "";
}

vector<int> Environment::validActions(int agentID){
    if(agentID != currAgent){
        return vector<int>{0};
    }
    vector<int> actions;
    int minAction = 0;
    if(timeIndex > 0){
        minAction = bets[timeIndex-1] + 1;
    }
    for(int i=minAction; i<2*diceSize; i++){
        actions.push_back(i);
    }
    if(timeIndex > 0){
        actions.push_back(CALL_ACTION);
    }
    return actions;
}

void Environment::makeAction(int* action, double* reward){
    for(int i=0; i<numAgents; i++){
        reward[i] = 0;
    }
    if(action[currAgent] == CALL_ACTION){
        endState = true;
        int lastBet = bets[timeIndex-1];
        int lastBetValue = lastBet % diceSize;
        int lastBetCount = (lastBet / diceSize) + 1;
        int count = 0;
        for(int i=0; i<numAgents; i++){
            if(diceVals[i] == lastBetValue || diceVals[i] == diceSize-1){
                count ++;
            }
        }
        reward[currAgent] = (count < lastBetCount) - (count >= lastBetCount);
        reward[1-currAgent] = -reward[currAgent];
    }
    else{
        bets[timeIndex] = action[currAgent];
        currAgent = 1-currAgent;
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
    for(int i=0; i<timeIndex; i++){
        features[bets[i] + (i%2)*2*diceSize] = 1;
    }
    features[diceVals[agentID] + 4*diceSize] = 1;
    features[agentID + 5*diceSize] = 1;
}