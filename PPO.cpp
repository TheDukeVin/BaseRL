
#include "PPO.h"

void PPOStore::empty(){
    index = 0;
}

void PPOStore::enqueue(PGInstance instance){
    assert(index < BufferSize);
    queue[index] = instance;
    index ++;
}

void PPOStore::shuffleQueue(){
    shuffle(queue, queue + index, default_random_engine{dev()});
}



PPO::PPO(LSTM::PVUnit* structure_, PPOStore* dataset_, string gameFile_, string saveFile_, string controlFile_, string scoreFile_){
    gameFile = gameFile_;
    saveFile = saveFile_;
    controlFile = controlFile_;
    scoreFile = scoreFile_;

    dataset = dataset_;

    structure = new LSTM::PVUnit(structure_, NULL);
    structure->randomize(0.1);
    structure->resetGradient();
    net = new LSTM::PVUnit(structure, NULL);
    net->copyParams(structure);
}

void PPO::rollout(bool print){
    Environment env;
    vector<PGInstance> trajectory;
    for(int t=0; t<timeHorizon; t++){
        vector<int> validActions = env.validActions();

        env.getFeatures(net->envInput->data);
        net->forwardPass();

        // Compute policy
        double policy[numActions];
        computeSoftmaxPolicy(net->policyOutput->data, numActions, validActions, policy);

        // Sample action
        int action = sampleDist(policy, numActions);

        PGInstance instance;
        instance.env = env;
        instance.action = action;
        for(int j=0; j<numActions; j++){
            instance.policy[j] = policy[j];
        }
        instance.reward = env.makeAction(action);
        trajectory.push_back(instance);


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

        dataset->enqueue(trajectory[t]);

        // Update value moments
        valueUpdateCount ++;
        valueFirstMoment = valueFirstMoment * (1-valueUpdateRate) + value * valueUpdateRate;
        valueSecondMoment = valueSecondMoment * (1-valueUpdateRate) + pow(value, 2) * valueUpdateRate;
        double weight = 1 - pow(1-valueUpdateRate, valueUpdateCount);
        if(valueNormOverride > 0){
            valueNorm = valueNormOverride;
        }
        else{
            valueNorm = sqrt(valueSecondMoment/weight - pow(valueFirstMoment/weight, 2)) * valueNormConstant;
        }
    }
    rolloutValue = total_reward;
    finalValue = trajectory[trajectory.size()-1].reward;
    rolloutTime = trajectory.size();
}

void PPO::generateDataset(int numRollouts){
    dataset->empty();
    for(int i=0; i<numRollouts; i++){
        rollout();

        sumScore += rolloutValue;
        lossCount += (finalValue < -2);
        winCount += (finalValue > 2);
        if(finalValue > 2){
            winTime += rolloutTime;
        }
    }
}

void PPO::accGrad(PGInstance instance){
    instance.env.getFeatures(net->envInput->data);
    net->forwardPass();
    net->resetGradient();
    for(int i=0; i<numActions; i++){
        net->policyOutput->gradient[i] = 0;
    }

    vector<int> validActions = instance.env.validActions();
    double currPolicy[numActions];
    computeSoftmaxPolicy(net->policyOutput->data, numActions, validActions, currPolicy);

    // Add PPO gradient
    double advantage = instance.value / valueNorm - net->valueOutput->data[0];
    double policyRatio = currPolicy[instance.action] / instance.policy[instance.action];
    if((advantage > 0 && policyRatio < 1 + clipRange) || (advantage < 0 && policyRatio > 1 - clipRange)){
        for(auto a : validActions){
            net->policyOutput->gradient[a] = (instance.policy[a] - (a == instance.action)) * advantage * policyRatio;
        }
    }

    // Add Entropy gradient
    double entropy = 0;
    for(auto a : validActions){
        entropy += instance.policy[a] * log(instance.policy[a]);
    }
    for(auto a : validActions){
        net->policyOutput->gradient[a] += instance.policy[a] * (log(instance.policy[a]) - entropy) * entropyConstant;
    }

    // Add Value gradient
    net->valueOutput->gradient[0] = net->valueOutput->data[0] - instance.value / valueNorm;

    net->backwardPass();
    structure->accumulateGradient(net);
}

void PPO::trainEpoch(int batchSize){
    structure->resetGradient();
    for(int i=0; i<dataset->index; i++){
        accGrad(dataset->queue[i]);
        if(i % batchSize == 0 && i > 0){
            structure->updateParams(alpha, -1, regRate);
            net->copyParams(structure);
        }
    }
}

void PPO::train(int numRollouts, int batchSize, int numEpochs, int numIter){
    assert(BufferSize >= numRollouts * (timeHorizon + 1));
    load();

    double evalSum = 0;

    int savePeriod = 100;
    int evalPeriod = 100;

    for(; iterationCount<=numIter; iterationCount++){
        generateDataset(numRollouts);
        for(int j=0; j<numEpochs; j++){
            dataset->shuffleQueue();
            trainEpoch(batchSize);
        }
        if(iterationCount > 0 && iterationCount % savePeriod == 0){
            save();
        }
        if(iterationCount > 0 && iterationCount % evalPeriod == 0){
            ofstream controlOut(controlFile, ios::app);
            controlOut << "Iteration " << iterationCount << " Time: " << (time(0) - start_time) << " Score: " 
                    << (sumScore / numRollouts / evalPeriod) << " Loss: " << (lossCount / numRollouts / evalPeriod) << " Win: " << (winCount / numRollouts / evalPeriod) 
                    << " valueNorm: " << valueNorm;
            if(winCount > 0.5){
                controlOut << " Win Time: " << (winTime / winCount);
            }
            controlOut << '\n';
            if(iterationCount > numIter/2){
                evalSum += sumScore / numRollouts;
            }
            sumScore = lossCount = winCount = winTime = 0;
        }
    }

    {
        ofstream controlOut(controlFile, ios::app);
        controlOut << "Evaluation score: " << (evalSum / (numIter/2)) << '\n';
    }
}

void PPO::save(){
    ofstream fout (saveFile);
    fout << iterationCount << '\n';
    fout << start_time << ' ' << valueFirstMoment << ' ' << valueSecondMoment << ' ' << valueUpdateCount << '\n';
    fout << structure->save() << '\n';
}

void PPO::load(){
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
    iterationCount = stoi(firstLine);
    string valueNormInfo;
    getline(fin, valueNormInfo);
    stringstream sin(valueNormInfo);
    sin >> start_time >> valueFirstMoment >> valueSecondMoment >> valueUpdateCount;
    string networkSave;
    getline(fin, networkSave);
    structure->load(networkSave);
    net->copyParams(structure);
}