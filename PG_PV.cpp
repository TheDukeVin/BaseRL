
#include "PG.h"

PG_PV::PG_PV(LSTM::PVUnit* structure_, string gameOutFile_){
    gameOutFile = gameOutFile_;
    ofstream gameOut (gameOutFile_);
    gameOut.close();

    structure = new LSTM::PVUnit(structure_, NULL);
    structure->randomize(0.1);
    structure->resetGradient();
    net = new LSTM::PVUnit(structure, NULL);
    net->copyParams(structure);
}

void PG_PV::rollout(bool print){
    Environment env;
    vector<PGInstance> trajectory;
    for(int t=0; t<timeHorizon; t++){
        vector<int> validActions = env.validActions();

        PGInstance instance;
        env.getFeatures(net->envInput->data);
        net->forwardPass();

        // Compute policy
        double policy[numActions];
        computeSoftmaxPolicy(net->policyOutput->data, numActions, validActions, policy);
        for(int j=0; j<numActions; j++){
            instance.policy[j] = policy[j];
        }

        // Sample action
        int action;
        if(randUniform() < epsilon){
            action = validActions[randomN(validActions.size())];
        }
        else{
            action = sampleDist(policy, numActions);
        }
        instance.action = action;

        instance.env = env;
        trajectory.push_back(instance);

        trajectory[t].reward = env.makeAction(action);

        if(print){
            ofstream gameOut (gameOutFile, ios::app);
            gameOut << trajectory[t].env.toString();
            gameOut << "Policy: ";
            unordered_map<int, double> policyMap;
            for(auto a : validActions){
                policyMap[a] = policy[a];
            }
            for(int i=0; i<numActions; i++){
                if(policyMap.find(i) == policyMap.end()){
                    gameOut << ". ";
                }
                else{
                    gameOut << policyMap[i] << ' ';
                }
            }
            gameOut << '\n';
            gameOut << "Value: " << net->valueOutput->data[0] * valueNorm << '\n';
            gameOut << "Action: " << action << " Reward: " << trajectory[t].reward << "\n\n";
        }
        if(env.endState) break;
    }

    // For Snake, we use the Value network to estimate the additional value if we continue the game
    double value = 0;
    if(env.validActions().size() != 0){
        env.getFeatures(net->envInput->data);
        net->forwardPass();
        value = net->valueOutput->data[0] * valueNorm;
    }

    double total_reward = 0;
    for(int t=trajectory.size()-1; t>=0; t--){
        value *= discountFactor;
        value += trajectory[t].reward;
        total_reward += trajectory[t].reward;
        trajectory[t].value = value;

        accGrad(trajectory[t]);

        // Update value moments
        valueUpdateCount ++;
        valueFirstMoment = valueFirstMoment * (1-valueUpdateRate) + value * valueUpdateRate;
        valueSecondMoment = valueSecondMoment * (1-valueUpdateRate) + pow(value, 2) * valueUpdateRate;
        double weight = 1 - pow(valueUpdateRate, valueUpdateCount);
        valueNorm = sqrt(valueSecondMoment/weight - pow(valueFirstMoment/weight, 2)) * valueNormConstant;
    }
    rolloutValue = total_reward;
    finalValue = trajectory[trajectory.size()-1].reward;
    rolloutTime = trajectory.size();
}

void PG_PV::accGrad(PGInstance instance){
    instance.env.getFeatures(net->envInput->data);
    net->forwardPass();
    net->resetGradient();
    for(int i=0; i<numActions; i++){
        net->policyOutput->gradient[i] = 0;
    }
    vector<int> validActions = instance.env.validActions();
    for(auto a : validActions){
        net->policyOutput->gradient[a] = (instance.policy[a] - (a == instance.action)) * (instance.value / valueNorm - net->valueOutput->data[0]);
    }
    double entropy = 0;
    for(auto a : validActions){
        entropy += instance.policy[a] * log(instance.policy[a]);
    }
    for(auto a : validActions){
        net->policyOutput->gradient[a] += instance.policy[a] * (log(instance.policy[a]) - entropy) * entropyConstant;
    }
    net->valueOutput->gradient[0] = net->valueOutput->data[0] - instance.value / valueNorm;
    net->backwardPass();
    structure->accumulateGradient(net);
}

void PG_PV::train(int batchSize, int numIter){
    unsigned start_time = time(0);

    ofstream fout("score.out");

    double sum = 0;
    double lossCount = 0;
    double winCount = 0;
    double winTime = 0;

    int evalPeriod = 1000;
    double evalSum = 0;

    string controlLog = "control.out";
    {
        ofstream controlOut ("control.out");
        controlOut.close();
    }
    for(int it=0; it<numIter; it++){
        for(int i=0; i<batchSize; i++){
            rollout();
            sum += rolloutValue;
            lossCount += (finalValue < -2);
            winCount += (finalValue > 2);
            if(finalValue > 2){
                winTime += rolloutTime;
            }
            if(it >= numIter/2){
                evalSum += rolloutValue;
            }
        }
        structure->updateParams(alpha, -1, regRate);
        net->copyParams(structure);
        if(it % evalPeriod == 0){
            if(it > 0){
                fout << ',';
            }
            double avgScore = sum / batchSize / evalPeriod;
            fout << avgScore;
            {
                ofstream controlOut(controlLog, ios::app);
                controlOut << "Iteration " << it << " Time: " << (time(0) - start_time) << ' ' << avgScore << " Loss: " << (lossCount / evalPeriod) << " Win: " << (winCount / evalPeriod) << " valueNorm: " << valueNorm;
                if(winCount > 0.5){
                    controlOut << " Win Time: " << (winTime / winCount);
                }
                controlOut << '\n';
                controlOut.close();
            }
            sum = 0;
            lossCount = 0;
            winCount = 0;
            winTime = 0;
        }
    }
    fout << '\n';
    {
        ofstream controlOut(controlLog, ios::app);
        controlOut << "Evaluation score: " << (evalSum / batchSize / (numIter/2)) << '\n';
        controlOut.close();
    }
}