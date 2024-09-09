
#include "MAPG.h"

MAPG::MAPG(LSTM::PVUnit* structure_, string gameFile_, string saveFile_, string controlFile_, string scoreFile_, string firstFile_){
    gameFile = gameFile_;
    saveFile = saveFile_;
    controlFile = controlFile_;
    scoreFile = scoreFile_;
    firstFile = firstFile_;

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

void MAPG::rollout(bool print){
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
            for(int j=0; j<numActions; j++){
                instance.policy[i][j] = policy[j];
                // cout << policy[j] << ' ';
            }
            // cout << "\n\n";


            // Sample action
            int action = sampleDist(policy, numActions);
            instance.action[i] = action;

            networkValues[i][t] = net[i][t]->valueOutput->data[0] * valueNorm;
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

void MAPG::train(int batchSize, int numIter, int evalPeriod, int savePeriod, int numEval, double alpha_){
    // cout << "Starting training session\n";
    assert(numAgents == 2);
    alpha = alpha_;
    unsigned start_time = time(0);

    // cout << "Loading data\n";

    // Load data
    load();

    // cout << "Filling tournament\n";

    int numImage = (numIter / evalPeriod) + 1;
    double comp[numImage][numImage];
    for(int i=0; i<numImage; i++){
        for(int j=0; j<numImage; j++){
            comp[i][j] = 0;
        }
    }

    // cout << "Running iterations\n";
    
    for(; iterationCount<=numIter+1; iterationCount++){
        // cout << "Generate rollouts\n";
        for(int i=0; i<batchSize; i++){
            rolloutID = i;
            rollout();
        }
        // cout << "Update params\n";
        structure->updateParams(alpha, -1, regRate);
        for(int ag=0; ag<numAgents; ag++){
            for(int i=0; i<timeHorizon; i++){
                net[ag][i]->copyParams(structure);
            }
        }
        // cout << "Evaluate\n";
        if(iterationCount % evalPeriod == 0){
            // Add network to evaluation.
            LSTM::PVUnit* image = new LSTM::PVUnit(structure, NULL);
            image->copyParams(structure);

            for(int i=0; i<netImages.size(); i++){
                // Compare current image to past images
                double sumRewards = 0;
                for(int j=0; j<numEval; j++){
                    if(j % 2 == 0){
                        for(int t=0; t<timeHorizon; t++){
                            net[0][t]->copyParams(structure);
                            net[1][t]->copyParams(netImages[i]);
                        }
                        rollout();
                        sumRewards += totalReward[0];
                    }
                    else{
                        for(int t=0; t<timeHorizon; t++){
                            net[0][t]->copyParams(netImages[i]);
                            net[1][t]->copyParams(structure);
                        }
                        rollout();
                        sumRewards += totalReward[1];
                    }
                    structure->resetGradient();
                }
                    
                comp[i][netImages.size()] = sumRewards / numEval;
            }
            netImages.push_back(image);
            string s = "";
            for(int i=0; i<netImages.size(); i++){
                for(int j=0; j<netImages.size(); j++){
                    s += to_string(comp[i][j]) + ' ';
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
    fout << iterationCount << ' ' << start_time << '\n';
    fout << structure->save() << '\n';
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
        return;
    }
    stringstream sin(firstLine);
    sin >> iterationCount >> start_time;
    string networkSave;
    getline(fin, networkSave);
    structure->load(networkSave);
    for(int i=0; i<numAgents; i++){
        for(int t=0; t<timeHorizon; t++){
            net[i][t]->copyParams(structure);
        }
    }
}