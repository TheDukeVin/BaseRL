
#include "policy.h"

Policy::Policy(int numIn, int numOut, LSTM::Model structure_){
    numInputs = numIn;
    numOutputs = numOut;
    features = new double[numInputs];
    outputPolicy = new double[numOutputs];

    structure = structure_;
    structure.randomize(0.1);
    structure.resetGradient();
    netInput = new LSTM::Data(structure.inputSize);
    netOutput = new LSTM::Data(structure.outputSize);
    net = LSTM::Model(&structure, NULL, netInput, netOutput);
}

void Policy::evaluate(vector<int> validActions){
    if(validActions.size() == 1){
        for(int i=0; i<numOutputs; i++){
            outputPolicy[i] = -1;
        }
        for(auto a : validActions){
            outputPolicy[a] = (a == validActions[0]);
        }
        return;
    }
    for(int i=0; i<numInputs; i++){
        netInput->data[i] = features[i];
    }
    net.forwardPass();
    computeSoftmaxPolicy(netOutput->data, numOutputs, validActions, outputPolicy);
}

void Policy::resetGrad(){
    structure.resetGradient();
}

void Policy::accGrad(vector<int> validActions, int action, double value){
    if(validActions.size() == 1){
        return;
    }
    evaluate(validActions);
    net.resetGradient();
    double entropy = 0;
    for(auto a : validActions){
        entropy += outputPolicy[a] * log(outputPolicy[a]);
    }
    for(int i=0; i<numOutputs; i++){
        netOutput->gradient[i] = 0;
    }
    for(auto a : validActions){
        netOutput->gradient[a] = 0.001 * netOutput->data[a] + outputPolicy[a] * (log(outputPolicy[a]) - entropy) * entropyConstant + (outputPolicy[a] - (a == action)) * value;
    }
    net.backwardPass();
    structure.accumulateGradient(&net);
}

void Policy::update(double scale, double momentum){
    structure.updateParams(scale, momentum, 1e-05);
    net.copyParams(&structure);
}