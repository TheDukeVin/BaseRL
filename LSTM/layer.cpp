
#include "lstm.h"

using namespace LSTM;

Data* Layer::addData(int size){
    Data* data = new Data(size);
    allHiddenData.push_back(data);
    return data;
}

void Layer::forwardPass(){
    resetGradient();
    for(int i=0; i<allNodes.size(); i++){
        allNodes[i]->forwardPass();
        // cout << "I1 ";
        // for(int j=0; j<allNodes[i]->i1->size; j++){
        //     cout << allNodes[i]->i1->data[j] << ' ';
        // }
        // cout << '\n';
        // if(allNodes[i]->i2){
        //     cout << "I2 ";
        //     for(int j=0; j<allNodes[i]->i2->size; j++){
        //         cout << allNodes[i]->i2->data[j] << ' ';
        //     }
        //     cout << '\n';
        // }
        // cout << "O ";
        // for(int j=0; j<allNodes[i]->o->size; j++){
        //     cout << allNodes[i]->o->data[j] << ' ';
        // }
        // cout << '\n';
    }
}

void Layer::backwardPass(){
    for(int i=allNodes.size()-1; i>=0; i--){
        allNodes[i]->backwardPass();
    }
}

void Layer::resetGradient(){
    input->resetGradient();
    output->resetGradient();
    for(int i=0; i<allHiddenData.size(); i++){
        allHiddenData[i]->resetGradient();
    }
    params->resetGradient();
}

void Layer::copyAct(Layer* l){
    input->copy(l->input);
    output->copy(l->output);
    for(int i=0; i<allHiddenData.size(); i++){
        allHiddenData[i]->copy(l->allHiddenData[i]);
    }
}