
#include "MAPG.h"

MAPG::MAPG(LSTM::PVUnit* structure_, string gameFile_, string saveFile_, string controlFile_, string scoreFile_, string firstFile_, string hiddenFile_){
    gameFile = gameFile_;
    saveFile = saveFile_;
    controlFile = controlFile_;
    scoreFile = scoreFile_;
    firstFile = firstFile_;
    hiddenFile = hiddenFile_;

    structure = new LSTM::PVUnit(structure_, NULL);
    structure->randomize(0.1);
    structure->resetGradient();

    for(int i=0; i<numAgents; i++){
        for(int t=0; t<timeHorizon; t++){
            LSTM::PVUnit* prevUnit;

            if(t>0) prevUnit = net[i][t-1];
            else prevUnit = NULL;

            net[i][t] = new LSTM::PVUnit(structure, prevUnit);
            net[i][t]->copyParams(structure);
        }
    }
    
}

void MAPG::rollout(bool train, bool print, bool outputHiddenVals){
    Environment env;
    vector<PGInstance> trajectory;

    double networkValues[numAgents][timeHorizon+1];

    for(int t=0; t<timeHorizon; t++){
        PGInstance instance;
        for(int i=0; i<numAgents; i++){
            // if(env.validActions(i).size() == 1){
            //     instance.policy[i][0] = 1;
            //     instance.action[i] = 0;
            //     continue;
            // }

            env.getFeatures(i, net[i][t]->envInput->data);
            net[i][t]->forwardPass();

            // Compute policy
            double policy[numActions];
            computeSoftmaxPolicy(net[i][t]->policyOutput->data, numActions, env.validActions(i), policy);
            // cout << "Policy: ";
            for(int j=0; j<numActions; j++){
                instance.policy[i][j] = policy[j];
                // cout << policy[j] << ' ';
            }
            // cout << "\n\n";


            // Sample action
            int action = sampleDist(policy, numActions);
            instance.action[i] = action;

            networkValues[i][t] = net[i][t]->valueOutput->data[0] * valueNorm;

            // Output hidden vals

            if(outputHiddenVals){
                ofstream hiddenOut (hiddenFile, ios::app);
                for(int j=0; j<net[i][t]->commonComp->size; j++){
                    hiddenOut << net[i][t]->commonComp->data[j] << ' ';
                }
                hiddenOut << env.currAgent << ' ' << env.toCode() << ' ' << env.timeIndex << ' ' << action << ' ' << networkValues[i][t];
                hiddenOut << '\n';
            }
        }

        instance.env = env;
        trajectory.push_back(instance);

        env.makeAction(instance.action, trajectory[t].reward);

        if(print){
            ofstream gameOut (gameFile, ios::app);
            gameOut << trajectory[t].env.toString();
            for(int i=0; i<numAgents; i++){
                gameOut << "Agent " << i << " Policy: ";
                unordered_map<int, double> policyMap;
                for(auto a : trajectory[t].env.validActions(i)){
                    policyMap[a] = instance.policy[i][a];
                }
                for(int j=0; j<numActions; j++){
                    if(policyMap.find(j) == policyMap.end()){
                        gameOut << ". ";
                    }
                    else{
                        gameOut << policyMap[j] << ' ';
                    }
                }
                gameOut << '\n';
                gameOut << "Value: " << networkValues[i][t] << '\n';
                gameOut << "Action: " << instance.action[i] << " Reward: " << trajectory[t].reward[i] << "\n\n";
            }
            

            ofstream codeOut("code.out", ios::app);
            codeOut << trajectory[t].env.toCode();
        }

        // Print first policies

        if(t == 0 && rolloutID == 0){
            ofstream fout(firstFile, ios::app);
            for(int i=0; i<numAgents; i++){
                for(int j=0; j<numActions; j++){
                    fout << instance.policy[i][j] << ' ';
                }
            }
            fout << '\n';
        }

        
        
        if(env.endState) break;
    }

    double value[numAgents];
    double advantage[numAgents];
    for(int i=0; i<numAgents; i++){
        value[i] = advantage[i] = networkValues[i][trajectory.size()] = totalReward[i] = 0;
    }

    // double value = 0;
    // networkValues[trajectory.size()] = 0;
    // double advantage = 0;

    if(!train) return;

    for(int t=trajectory.size()-1; t>=0; t--){
        for(int i=0; i<numAgents; i++){
            advantage[i] *= discountFactor*GAEParam;
            advantage[i] += trajectory[t].reward[i] + (discountFactor * networkValues[i][t+1]) - networkValues[i][t];
            trajectory[t].advantage[i] = advantage[i];

            value[i] *= discountFactor;
            value[i] += trajectory[t].reward[i];
            trajectory[t].value[i] = value[i];

            totalReward[i] += trajectory[t].reward[i];

            assert(abs(trajectory[t].advantage[i] - (trajectory[t].value[i] - networkValues[i][t])) < 1e-8);
        }
        accGrad(trajectory[t], t);
    }
}

void MAPG::accGrad(PGInstance instance, int index){
    for(int i=0; i<numAgents; i++){
        if(instance.env.validActions(i).size() == 1){
            continue;
        }

        for(int j=0; j<numActions; j++){
            net[i][index]->policyOutput->gradient[j] = 0;
        }
        vector<int> validActions = instance.env.validActions(i);
        for(auto a : validActions){
            net[i][index]->policyOutput->gradient[a] = (instance.policy[i][a] - (a == instance.action[i])) * (instance.value[i] / valueNorm - net[i][index]->valueOutput->data[0]);
        }
        double entropy = 0;
        for(auto a : validActions){
            entropy += instance.policy[i][a] * log(instance.policy[i][a]);
        }
        for(auto a : validActions){
            net[i][index]->policyOutput->gradient[a] += instance.policy[i][a] * (log(instance.policy[i][a]) - entropy) * entropyConstant;
        }
        net[i][index]->valueOutput->gradient[0] = net[i][index]->valueOutput->data[0] - instance.value[i] / valueNorm;
        net[i][index]->backwardPass();
        structure->accumulateGradient(net[i][index]);
    }
}

void MAPG::multRollout(int N){
    for(int i=0; i<numAgents; i++){
        multTotalReward[i] = 0;
    }
    for(int i=0; i<N; i++){
        rollout();
        for(int j=0; j<numAgents; j++){
            multTotalReward[j] += totalReward[j];
        }
        rolloutID ++;
    }
}

void MAPG::train(int batchSize, int numIter, int evalPeriod, int savePeriod, int numEval, double alpha_, double entropyConstant_, int numThreads){
    assert(numAgents == 2);
    alpha = alpha_;
    entropyConstant = entropyConstant_;
    unsigned start_time = time(0);

    // Load data
    load();

    // int numImage = (numIter / evalPeriod) + 1;
    // double comp[numImage][numImage];
    // for(int i=0; i<numImage; i++){
    //     for(int j=0; j<numImage; j++){
    //         comp[i][j] = 0;
    //     }
    // }

    MAPG* subtrainers[numThreads];
    thread* threads[numThreads];
    for(int i=0; i<numThreads; i++){
        subtrainers[i] = new MAPG(structure, gameFile, saveFile, controlFile, scoreFile, firstFile, hiddenFile);
        subtrainers[i]->entropyConstant = entropyConstant;
    }
    
    for(; iterationCount<=numIter+2; iterationCount++){
        // for(int i=0; i<batchSize; i++){
        //     rolloutID = i;
        //     rollout();
        // }
        for(int i=0; i<numThreads; i++){
            subtrainers[i]->structure->copyParams(structure);
            subtrainers[i]->structure->resetGradient();
            for(int ag=0; ag<numAgents; ag++){
                for(int t=0; t<timeHorizon; t++){
                    subtrainers[i]->net[ag][t]->copyParams(structure);
                }
            }
            subtrainers[i]->rolloutID = i*(batchSize/numThreads);
            threads[i] = new thread(&MAPG::multRollout, subtrainers[i], batchSize/numThreads);
        }
        for(int i=0; i<numThreads; i++){
            threads[i]->join();
            structure->accumulateGradient(subtrainers[i]->structure);
        }
        structure->updateParams(alpha, -1, regRate);
        // for(int ag=0; ag<numAgents; ag++){
        //     for(int t=0; t<timeHorizon; t++){
        //         net[ag][t]->copyParams(structure);
        //     }
        // }
        if(iterationCount % evalPeriod == 0){
            // Add network to evaluation.
            LSTM::PVUnit* image = new LSTM::PVUnit(structure, NULL);
            image->copyParams(structure);

            for(int i=0; i<netImages.size(); i++){
                // Compare current image to past images
                for(int j=0; j<numThreads; j++){
                    int activeAgent = j%2;
                    for(int t=0; t<timeHorizon; t++){
                        subtrainers[j]->net[activeAgent][t]->copyParams(structure);
                        subtrainers[j]->net[1-activeAgent][t]->copyParams(netImages[i]);
                    }
                    threads[j] = new thread(&MAPG::multRollout, subtrainers[j], numEval/numThreads);
                    // if(j % 2 == 0){
                    //     for(int t=0; t<timeHorizon; t++){
                    //         net[0][t]->copyParams(structure);
                    //         net[1][t]->copyParams(netImages[i]);
                    //     }
                    //     rollout();
                    //     sumRewards += totalReward[0];
                    // }
                    // else{
                    //     for(int t=0; t<timeHorizon; t++){
                    //         net[0][t]->copyParams(netImages[i]);
                    //         net[1][t]->copyParams(structure);
                    //     }
                    //     rollout();
                    //     sumRewards += totalReward[1];
                    // }
                    // structure->resetGradient();
                }

                double sumRewards = 0;
                for(int j=0; j<numThreads; j++){
                    int activeAgent = j%2;
                    threads[j]->join();
                    sumRewards += subtrainers[j]->multTotalReward[activeAgent];
                }
                    
                evaluation[i].push_back(sumRewards / numEval);
            }
            evaluation.push_back(vector<double>());
            netImages.push_back(image);
            for(int i=0; i<netImages.size(); i++){
                evaluation[netImages.size()-1].push_back(0);
            }
            string s = "";
            for(int i=0; i<netImages.size(); i++){
                for(int j=0; j<netImages.size(); j++){
                    s += to_string(evaluation[i][j]) + ' ';
                }
                s += '\n';
            }

            ofstream fout(controlFile, ios::app);
            fout << "Iteration " << netImages.size() << '\n';
            fout << s;
            fout << "Time: " << (time(0) - start_time) << '\n';
            fout << '\n';

            ofstream scoreOut(scoreFile);
            scoreOut << s;
        }
        if(iterationCount > 0 && iterationCount % savePeriod == 0){
            save();
        }
    }
}

void MAPG::save(){
    ofstream fout (saveFile);
    fout << iterationCount << ' ' << start_time << ' ' << netImages.size() << '\n';
    fout << structure->save() << '\n';

    for(int i=0; i<netImages.size(); i++){
        fout << netImages[i]->save() << '\n';
    }

    for(int i=0; i<evaluation.size(); i++){
        for(int j=0; j<evaluation[0].size(); j++){
            fout << evaluation[i][j] << ' ';
        }
    }
    fout << '\n';
}

void MAPG::load(){
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
        {
            ofstream fout(firstFile);
            fout.close();
        }
        {
            ofstream fout(hiddenFile);
            fout.close();
        }
        return;
    }
    int numImages;
    {
        stringstream sin(firstLine);
        sin >> iterationCount >> start_time >> numImages;
        iterationCount ++;
    }
    {
        string networkSave;
        getline(fin, networkSave);
        structure->load(networkSave);
    }
    
    // for(int i=0; i<numAgents; i++){
    //     for(int t=0; t<timeHorizon; t++){
    //         net[i][t]->copyParams(structure);
    //     }
    // }
    for(int i=0; i<numImages; i++){
        string networkSave;
        getline(fin, networkSave);
        netImages.push_back(new LSTM::PVUnit(structure, NULL));
        netImages[i]->load(networkSave);
    }
    string evaluationSave;
    getline(fin, evaluationSave);
    {
        stringstream sin(evaluationSave);
        for(int i=0; i<numImages; i++){
            evaluation.push_back(vector<double>());
            for(int j=0; j<numImages; j++){
                double eval; sin >> eval;
                evaluation[i].push_back(eval);
            }
        }
    }
}