
#include "policy.h"

Policy::Policy(int numIn, int numOut){
    numInputs = numIn;
    numOutputs = numOut;
    weights = new double[numInputs * numOutputs];
    weightGrad = new double[numInputs * numOutputs];
    features = new double[numInputs];
    outputPolicy = new double[numOutputs];
    for(int i=0; i<numInputs*numOutputs; i++){
        weights[i] = 0;
        weightGrad[i] = 0;
    }
}

void Policy::evaluate(vector<int> validActions){
    double logits[numOutputs];
    for(int j=0; j<numOutputs; j++){
        double sum = 0;
        for(int i=0; i<numInputs; i++){
            sum += weights[i*numOutputs + j] * features[i];
        }
        logits[j] = sum;
    }
    computeSoftmaxPolicy(logits, numOutputs, validActions, outputPolicy);
}

void Policy::resetGrad(){
    for(int i=0; i<numInputs*numOutputs; i++){
        weightGrad[i] = 0;
    }
}

void Policy::accGrad(vector<int> validActions, int action, double value){
    evaluate(validActions);
    double Dlogits[numOutputs];
    for(int i=0; i<numOutputs; i++){
        Dlogits[i] = 0;
    }
    double entropy = 0;
    for(auto a : validActions){
        entropy += outputPolicy[a] * log(outputPolicy[a]);
    }
    for(auto a : validActions){
        Dlogits[a] = outputPolicy[a] * (log(outputPolicy[a]) - entropy) * entropyConstant + (outputPolicy[a] - (a == action)) * value;
    }
    for(int i=0; i<numInputs; i++){
        for(int j=0; j<numOutputs; j++){
            weightGrad[i*numOutputs + j] += features[i] * Dlogits[j];
        }
    }
}

void Policy::update(double scale){
    for(int i=0; i<numInputs*numOutputs; i++){
        weights[i] -= weightGrad[i] * scale;
    }
}