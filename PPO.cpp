
#include "PPO.h"

void PPOStore::empty(){
    index = 0;
}

void PPOStore::enqueue(PGInstance instance){
    queue[index%BufferSize] = instance;
    index ++;
}

void PPOStore::shuffleQueue(){
    shuffle(queue, queue + getSize(), default_random_engine{dev()});
}

int PPOStore::getSize(){
    return min(index, BufferSize);
}


PPO::PPO(LSTM::PVUnit* structure_, PPOStore* dataset_, string gameFile_, string saveFile_, string controlFile_, string scoreFile_){
    gameFile = gameFile_;
    saveFile = saveFile_;
    controlFile = controlFile_;
    scoreFile = scoreFile_;

    dataset = dataset_;

    symNet = SymUnit(structure_);
}

void PPO::rollout(bool print){
    Environment env;
    vector<PGInstance> trajectory;
    double networkValues[timeHorizon+1];
    for(int t=0; t<timeHorizon; t++){
        vector<int> validActions = env.validActions();

        symNet.forwardPass(env, randomSym());

        // Compute policy
        double policy[numActions];
        computeSoftmaxPolicy(symNet.policyOutput->data, numActions, validActions, policy);

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

        networkValues[t] = symNet.valueOutput->data[0] * valueNorm;

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
            gameOut << "Network Value: " << symNet.valueOutput->data[0] * valueNorm << '\n';
            gameOut << "Potential: " << trajectory[t].env.potential() << '\n';
            gameOut << "Action: " << action << " Reward: " << trajectory[t].reward << "\n\n";
        }
        // assert(instance.reward >= 0);
        if(env.endState) break;
    }

    int valueRange = trajectory.size();

    // end value is negative potential function, in accordance with reward shaping.
    double value = -env.potential();

    // For Snake, we use the Value network to estimate the additional value if we continue the game
    if(env.validActions().size() != 0){
        assert(trajectory.size() == timeHorizon);
        symNet.forwardPass(env, randomSym());

        networkValues[trajectory.size()] = symNet.valueOutput->data[0] * valueNorm;
        value = networkValues[trajectory.size()];
        valueRange ++;
    }

    double advantage = 0;
    double total_reward = 0;
    for(int t=trajectory.size()-1; t>=0; t--){
        advantage *= discountFactor*GAEParam;
        advantage += trajectory[t].reward - networkValues[t];
        if(t+1 < valueRange){
            advantage += discountFactor * networkValues[t+1];
        }
        trajectory[t].advantage = advantage;
        trajectory[t].importance_weight = pow(1 + abs(advantage), importance_temp);

        value *= discountFactor;
        value += trajectory[t].reward;
        trajectory[t].value = value;

        // assert(abs(advantage - (value - networkValues[t])) < 1e-10);


        // dataset->enqueue(trajectory[t]);
        // Single-action optimization
        if(trajectory[t].env.validActions().size() > 1){
            dataset->enqueue(trajectory[t]);
        }

        total_reward += trajectory[t].reward;

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
    rolloutTime = trajectory.size();

    sumScore += total_reward;
    sumSize += env.snake.size;
    lossCount += (env.validActions().size() == 0 && env.snake.size < boardSize);
    if(env.snake.size == boardSize){
        winCount ++;
        winTime += rolloutTime;
    }
    gameCount ++;
}

void PPO::generateDataset(int numRollouts, int batchSize){
    for(int i=0; i<numRollouts || dataset->getSize() < batchSize; i++){
        rollout();
    }
}

void PPO::accGrad(PGInstance instance){
    symNet.forwardPass(instance.env, randomSym());
    for(int i=0; i<numActions; i++){
        symNet.policyOutput->gradient[i] = 0;
    }

    vector<int> validActions = instance.env.validActions();
    double currPolicy[numActions];
    computeSoftmaxPolicy(symNet.policyOutput->data, numActions, validActions, currPolicy);

    // Add PPO gradient

    double advantage = instance.advantage / valueNorm;
    double policyRatio = currPolicy[instance.action] / instance.policy[instance.action];
    if((advantage > 0 && policyRatio < 1 + clipRange) || (advantage < 0 && policyRatio > 1 - clipRange)){
        for(auto a : validActions){
            symNet.policyOutput->gradient[a] = (instance.policy[a] - (a == instance.action)) * advantage * policyRatio;
        }
    }

    // Add Entropy gradient
    double entropy = 0;
    for(auto a : validActions){
        entropy += instance.policy[a] * log(instance.policy[a]);
    }
    for(auto a : validActions){
        symNet.policyOutput->gradient[a] += instance.policy[a] * (log(instance.policy[a]) - entropy) * entropyConstant;
    }

    // Add Value gradient
    symNet.valueOutput->gradient[0] = symNet.valueOutput->data[0] - instance.value / valueNorm;

    // Use importance sampling

    for(auto a : validActions){
        symNet.policyOutput->gradient[a] /= instance.importance_weight;
    }
    symNet.valueOutput->gradient[0] /= instance.importance_weight;

    symNet.backwardPass();
}

void PPO::trainEpoch(int batchSize, bool pool){
    if(pool){
        double importance_dist[dataset->getSize()];
        int samples[batchSize];
        double sum = 0;
        for(int i=0; i<dataset->getSize(); i++){
            importance_dist[i] = dataset->queue[i].importance_weight;
            sum += importance_dist[i];
        }
        for(int i=0; i<dataset->getSize(); i++){
            importance_dist[i] /= sum;
        }
        sampleMult(importance_dist, dataset->getSize(), samples, batchSize);
        for(int i=0; i<batchSize; i++){
            accGrad(dataset->queue[samples[i]]);
        }
        symNet.update(alpha, regRate);
    }
    else{
        if(batchSize == NO_BATCH){
            for(int i=0; i<dataset->index; i++){
                accGrad(dataset->queue[i]);
            }
            symNet.update(alpha, regRate);
        }
        else{
            for(int i=0; i<dataset->index; i++){
                accGrad(dataset->queue[i]);
                if(i % batchSize == 0 && i > 0){
                    symNet.update(alpha, regRate);
                }
            }
        }
    }
    
}

double PPO::train(int numRollouts, int batchSize, int numEpochs, int numIter, int evalPeriod, int savePeriod, double alpha_, double GAEParam_, double importance_temp_, bool pool){
    alpha = alpha_;
    GAEParam = GAEParam_;
    importance_temp = importance_temp_;
    if(!pool) assert(BufferSize >= numRollouts * (timeHorizon + 1));
    load();

    double evalSum = 0;

    {
        ofstream fout(scoreFile, ios::app);
        fout << 0;
    }

    {
        ofstream controlOut(controlFile, ios::app);
        string s = "numRollouts: " + to_string(numRollouts) + " " +
               "batchSize: " + to_string(batchSize) + " " +
               "numEpochs: " + to_string(numEpochs) + " " +
               "numIter: " + to_string(numIter) + " " +
               "evalPeriod: " + to_string(evalPeriod) + " " +
               "savePeriod: " + to_string(savePeriod) + " " +
               "alpha: " + to_string(alpha) + " " +
               "GAEParam: " + to_string(GAEParam) + " " +
               "importance_temp: " + to_string(importance_temp) + " ";
        controlOut << "Hyperparameters: " << s << '\n';
    }

    for(; iterationCount<=numIter; iterationCount++){
        if(!pool){
            dataset->empty();
            generateDataset(numRollouts, batchSize);
            for(int j=0; j<numEpochs; j++){
                dataset->shuffleQueue();
                trainEpoch(batchSize, pool);
            }
        }
        else{
            generateDataset(numRollouts, 0);
            dataset->shuffleQueue();
            trainEpoch(batchSize, pool);
        }
        
        if(iterationCount > 0 && iterationCount % savePeriod == 0){
            save();
        }
        if(iterationCount > 0 && iterationCount % evalPeriod == 0){
            ofstream controlOut(controlFile, ios::app);
            controlOut << "Iteration " << iterationCount << " Time: " << (time(0) - start_time) << " Score: " 
                    << (sumScore / gameCount) << " Size: " << (sumSize / gameCount) << " Loss: " << (lossCount / gameCount) << " Win: " << (winCount / gameCount) 
                    << " valueNorm: " << valueNorm << " stepSize: " << alpha;
            if(winCount > 0.5){
                controlOut << " Win Time: " << (winTime / winCount);
            }
            controlOut << '\n';
            if(iterationCount > numIter/2){
                evalSum += sumScore / gameCount;
            }
            {
                ofstream fout(scoreFile, ios::app);
                fout << ',' << (sumScore / gameCount);
            }
            gameCount = sumSize = sumScore = lossCount = winCount = winTime = 0;
        }
    }

    {
        ofstream fout(scoreFile, ios::app);
        fout << '\n';
    }

    double evalScore = evalSum / (numIter/evalPeriod/2);

    {
        ofstream controlOut(controlFile, ios::app);
        controlOut << "Evaluation score: " << evalScore << '\n';
    }

    return evalScore;
}

void PPO::save(){
    ofstream fout (saveFile);
    fout << iterationCount << '\n';
    fout << start_time << ' ' << valueFirstMoment << ' ' << valueSecondMoment << ' ' << valueUpdateCount << '\n';
    fout << symNet.structure->save() << '\n';
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
    symNet.structure->load(networkSave);
    symNet.net->copyParams(symNet.structure);
}