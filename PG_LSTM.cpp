
#include "PG.h"

PG_LSTM::PG_LSTM(LSTM::PVUnit* structure_, string gameFile_, string saveFile_, string controlFile_, string scoreFile_){
    gameFile = gameFile_;
    saveFile = saveFile_;
    controlFile = controlFile_;
    scoreFile = scoreFile_;

    structure = new LSTM::PVUnit(structure_, NULL);
    structure->randomize(0.1);
    structure->resetGradient();

    for(int i=0; i<timeHorizon; i++){
        LSTM::PVUnit* prevUnit;

        if(i>0) prevUnit = net[i-1];
        else prevUnit = NULL;

        net[i] = new LSTM::PVUnit(structure, prevUnit);
        net[i]->copyParams(structure);
    }
}

void PG_LSTM::rollout(bool print){
    Environment env;
    vector<PGInstance> trajectory;

    double networkValues[timeHorizon+1];

    for(int t=0; t<timeHorizon; t++){
        vector<int> validActions = env.validActions();

        PGInstance instance;
        env.getFeatures(net[t]->envInput->data);
        net[t]->forwardPass();

        // Compute policy
        double policy[numActions];
        computeSoftmaxPolicy(net[t]->policyOutput->data, numActions, validActions, policy);
        for(int j=0; j<numActions; j++){
            instance.policy[j] = policy[j];
        }

        // Sample action
        int action = sampleDist(policy, numActions);
        instance.action = action;

        instance.env = env;
        trajectory.push_back(instance);

        trajectory[t].reward = env.makeAction(action);
        networkValues[t] = net[t]->valueOutput->data[0] * valueNorm;

        if(print){
            ofstream gameOut (gameFile, ios::app);
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
            gameOut << "Value: " << net[t]->valueOutput->data[0] * valueNorm << '\n';
            gameOut << "Action: " << action << " Reward: " << trajectory[t].reward << "\n\n";

            ofstream codeOut("code.out", ios::app);
            codeOut << trajectory[t].env.toCode();
        }
        if(env.endState) break;
    }

    double value = 0;
    networkValues[trajectory.size()] = 0;
    double advantage = 0;

    double total_reward = 0;
    for(int t=trajectory.size()-1; t>=0; t--){
        advantage *= discountFactor*GAEParam;
        advantage += trajectory[t].reward + (discountFactor * networkValues[t+1]) - networkValues[t];
        trajectory[t].advantage = advantage;

        value *= discountFactor;
        value += trajectory[t].reward;

        total_reward += trajectory[t].reward;
        trajectory[t].value = value;

        accGrad(trajectory[t], t);
    }
    rolloutValue = total_reward;
}

void PG_LSTM::accGrad(PGInstance instance, int index){
    for(int i=0; i<numActions; i++){
        net[index]->policyOutput->gradient[i] = 0;
    }
    vector<int> validActions = instance.env.validActions();
    for(auto a : validActions){
        net[index]->policyOutput->gradient[a] = (instance.policy[a] - (a == instance.action)) * (instance.value / valueNorm - net[index]->valueOutput->data[0]);
    }
    double entropy = 0;
    for(auto a : validActions){
        entropy += instance.policy[a] * log(instance.policy[a]);
    }
    for(auto a : validActions){
        net[index]->policyOutput->gradient[a] += instance.policy[a] * (log(instance.policy[a]) - entropy) * entropyConstant;
    }
    net[index]->valueOutput->gradient[0] = net[index]->valueOutput->data[0] - instance.value / valueNorm;
    net[index]->backwardPass();
    structure->accumulateGradient(net[index]);
}

void PG_LSTM::train(int batchSize, int numIter, int evalPeriod, int savePeriod, double alpha_){
    alpha = alpha_;
    unsigned start_time = time(0);

    double sum = 0;
    double evalSum = 0;

    // Load data
    load();
    
    for(; iterationCount<=numIter; iterationCount++){
        for(int i=0; i<batchSize; i++){
            rollout();
            sum += rolloutValue;
            if(iterationCount >= numIter/2){
                evalSum += rolloutValue;
            }
        }
        structure->updateParams(alpha, -1, regRate);
        for(int i=0; i<timeHorizon; i++){
            net[i]->copyParams(structure);
        }
        if(iterationCount % evalPeriod == 0){
            double avgScore = sum / batchSize / evalPeriod;
            {
                ofstream scoreOut (scoreFile, ios::app);
                if(iterationCount > 0){
                    scoreOut << ',';
                }
                scoreOut << avgScore;
            }
            
            {
                ofstream controlOut(controlFile, ios::app);
                controlOut << "Iteration " << iterationCount << " Time: " << (time(0) - start_time) << ' ' << avgScore << '\n';
                controlOut.close();
            }
            sum = 0;
        }
        if(iterationCount > 0 && iterationCount % savePeriod == 0){
            save();
        }
    }

    {
        ofstream scoreOut (scoreFile, ios::app);
        scoreOut << '\n';
    }

    {
        ofstream controlOut(controlFile, ios::app);
        controlOut << "Evaluation score: " << (evalSum / batchSize / (numIter/2)) << '\n';
        controlOut.close();
    }
}

void PG_LSTM::save(){
    ofstream fout (saveFile);
    fout << iterationCount << ' ' << start_time << '\n';
    fout << structure->save() << '\n';
}

void PG_LSTM::load(){
    ifstream fin (saveFile);
    string firstLine;
    if(!getline(fin, firstLine)){
        iterationCount = 0;
        start_time = time(0);
        {
            ofstream fout(gameFile);
            fout.close();
        }
        {
            ofstream fout(saveFile);
            fout.close();
        }
        {
            ofstream fout(controlFile);
            fout << "Training control Time: " << start_time << '\n';
            fout.close();
        }
        {
            ofstream fout(scoreFile);
            fout.close();
        }
        return;
    }
    stringstream sin(firstLine);
    sin >> iterationCount >> start_time;
    string networkSave;
    getline(fin, networkSave);
    structure->load(networkSave);
    for(int i=0; i<timeHorizon; i++){
        net[i]->copyParams(structure);
    }
}