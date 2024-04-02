
#include "DQN.h"

ReplayInstance::ReplayInstance(Environment currState_, Environment nextState_, int action_, double reward_, double value_){
    currState = currState_;
    nextState = nextState_;
    action = action_;
    reward = reward_;
    value = value_;
}


ReplayBuffer::ReplayBuffer(int start_size){
    index = 0;
    size = start_size;
    fillSize = 0;
}

void ReplayBuffer::updateSize(int new_size){
    assert(size <= new_size);
    size = new_size;
}

void ReplayBuffer::enqueue(ReplayInstance instance){
    queue[index] = instance;
    index ++;
    fillSize = max(fillSize, index);
    if(index == size){
        shuffle(queue, queue + size, default_random_engine{dev()});
        index = 0;
    }
}

ReplayInstance ReplayBuffer::randomInstance(){
    return queue[randomN(fillSize)];
}


DQN::DQN(LSTM::Model structure_, string gameOutFile_, ReplayBuffer* buffer_){
    gameOutFile = gameOutFile_;
    buffer = buffer_;
    ofstream gameOut (gameOutFile_);
    gameOut.close();

    structure = structure_;
    structure.randomize(0.1);
    structure.resetGradient();
    netInput = new LSTM::Data(structure.inputSize);
    netOutput = new LSTM::Data(structure.outputSize);
    net = LSTM::Model(&structure, NULL, netInput, netOutput);
}

void DQN::rollout(int batchSize, bool print){
    Environment env;
    Environment trajectory[timeHorizon];
    int actions[timeHorizon];
    double rewards[timeHorizon];
    double total_reward = 0;
    int t;
    vector<ReplayInstance> replay;
    for(t=0; t<timeHorizon; t++){
        int action = -1;
        vector<int> validActions = env.validActions();
        if(validActions.size() == 1 || randUniform() < epsilon){
            action = validActions[randomN(validActions.size())];
        }
        else{
            env.getFeatures(netInput->data);
            net.forwardPass();
            double maxQ = -1e+10;
            for(auto a : validActions){
                // add eta exploration
                netOutput->data[a] += eta / (randUniform() + 1e-10);
                if(netOutput->data[a] > maxQ){
                    maxQ = netOutput->data[a];
                    action = a;
                }
            }
        }
        assert(action >= 0);
        trajectory[t] = env;
        actions[t] = action;
        rewards[t] = env.makeAction(actions[t]);
        total_reward += rewards[t];
        // replay.push_back(ReplayInstance(trajectory[t], env, actions[t], rewards[t], -1));
        buffer->enqueue(ReplayInstance(trajectory[t], env, actions[t], rewards[t], -1));

        for(int j=0; j<batchSize; j++){
            updateInstance(buffer->randomInstance());
        }
        structure.updateParams(alpha, -1, regRate);
        net.copyParams(&structure);

        if(print){
            ofstream gameOut (gameOutFile, ios::app);
            gameOut << trajectory[t].toString();
            gameOut << "Q function: ";
            unordered_map<int, double> Qvals;
            for(auto a : validActions){
                Qvals[a] = netOutput->data[a];
            }
            for(int i=0; i<numActions; i++){
                if(Qvals.find(i) == Qvals.end()){
                    gameOut << ". ";
                }
                else{
                    gameOut << Qvals[i] << ' ';
                }
            }
            gameOut << '\n';
            gameOut << "Action: " << actions[t] << " Reward: " << rewards[t] << "\n\n";
        }
        if(env.endState) break;
    }

    // double value = 0;
    // for(int i=replay.size()-1; i>=0; i--){
    //     value *= discountFactor;
    //     value += replay[i].reward;
    //     replay[i].value = value;
    //     buffer->enqueue(replay[i]);
    // }

    // int rolloutLength = t+1;
    // trajectory[rolloutLength-1] = env;
    // double value = 0;
    // for(t=rolloutLength-2; t>=0; t--){
    //     value *= discountFactor;
    //     value += rewards[t];
    //     buffer->enqueue(ReplayInstance(trajectory[t], trajectory[t+1], actions[t], rewards[t], value));
    // }
    rolloutValue = total_reward;
    finalValue = rewards[replay.size()-1];
}

void DQN::updateInstance(ReplayInstance instance){
    double expectedOutput;
    if(instance.nextState.endState){
        expectedOutput = instance.reward;
    }
    else{
        instance.nextState.getFeatures(netInput->data);
        net.forwardPass();
        double maxQ = -1e+10;
        for(auto a : instance.nextState.validActions()){
            if(netOutput->data[a] > maxQ){
                maxQ = netOutput->data[a];
            }
        }
        expectedOutput = instance.reward + discountFactor * maxQ;
    }
    // expectedOutput = instance.value;
    instance.currState.getFeatures(netInput->data);
    net.forwardPass();
    for(int i=0; i<numActions; i++){
        netOutput->gradient[i] = 0;
    }
    netOutput->gradient[instance.action] = 2 * (netOutput->data[instance.action] - expectedOutput);
    net.backwardPass();
    structure.accumulateGradient(&net);
}

void DQN::train(int batchSize, int numBatches, int numRollouts){
    ofstream fout("score.out");
    string controlLog = "control.out";
    {
        ofstream controlOut (controlLog);
        controlOut.close();
    }
    unsigned start_time = time(0);

    int evalPeriod = 1000;
    double sum = 0;
    double lossCount = 0;
    double winCount = 0;
    double evalSum = 0;
    for(int it=0; it<numRollouts; it++){
        rollout(batchSize);
        sum += rolloutValue;
        lossCount += (finalValue < -2);
        winCount += (finalValue > 2);
        if(it >= numRollouts/2){
            evalSum += rolloutValue;
        }
        // for(int i=0; i<numBatches; i++){
        //     for(int j=0; j<batchSize; j++){
        //         updateInstance(buffer->randomInstance());
        //     }
        //     structure.updateParams(alpha, -1, regRate);
        //     net.copyParams(&structure);
        // }
        if(it % evalPeriod == 0){
            if(it > 0){
                fout << ',';
            }
            double avgScore = sum / evalPeriod;
            fout << avgScore;
            {
                ofstream controlOut(controlLog, ios::app);
                controlOut << "Iteration " << it << " Time: " << (time(0) - start_time) << ' ' << avgScore << " Loss: " << (lossCount / evalPeriod) << " Win: " << (winCount / evalPeriod) << '\n';
                controlOut.close();
            }
            sum = 0;
            lossCount = 0;
            winCount = 0;
        }
        // Buffer expansion
        // if(it == numRollouts / 2){
        //     buffer->updateSize(1000000);
        //     {
        //         ofstream controlOut(controlLog, ios::app);
        //         controlOut << "Changed buffer size to " << buffer->size;
        //         controlOut.close();
        //     }
        // }
    }
    {
        ofstream controlOut(controlLog, ios::app);
        controlOut << "Evaluation score: " << (evalSum / (numRollouts/2)) << '\n';
        controlOut.close();
    }
}