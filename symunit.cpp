
#include "PG.h"

SymUnit::SymUnit(LSTM::PVUnit* structure_){
    structure = new LSTM::PVUnit(structure_, NULL);
    structure->randomize(0.1);
    structure->resetGradient();
    net = new LSTM::PVUnit(structure, NULL);
    net->copyParams(structure);

    valueOutput = new LSTM::Data(1);
    policyOutput = new LSTM::Data(numActions);
}

void SymUnit::forwardPass(Environment env, int symID_){
    symID = symID_;
    env.getFeatures(net->envInput->data, symID);
    net->forwardPass();
    valueOutput->data[0] = net->valueOutput->data[0];
    for(int i=0; i<numActions; i++){
        policyOutput->data[i] = net->policyOutput->data[symAction(i, symID)];
    }
    net->resetGradient();
}

void SymUnit::backwardPass(){
    net->valueOutput->gradient[0] = valueOutput->gradient[0];
    for(int i=0; i<numActions; i++){
        net->policyOutput->gradient[symAction(i, symID)] = policyOutput->gradient[i];
    }
    net->backwardPass();
    structure->accumulateGradient(net);
}

void SymUnit::update(){
    structure->updateParams(alpha, -1, regRate);
    net->copyParams(structure);
}