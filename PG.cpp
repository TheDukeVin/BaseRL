
#include "PG.h"

PG::PG(LSTM::Model structure_, string gameOutFile_){
    learner = Policy(numFeatures, numActions, structure_);
    gameOutFile = gameOutFile_;
    ofstream gameOut (gameOutFile_);
    gameOut.close();

    for(int i=0; i<timeHorizon; i++){
        baseline[i] = 0;
    }
}

double PG::rollout(bool print){
    Environment env;
    Environment trajectory[timeHorizon];
    int actions[timeHorizon];
    double rewards[timeHorizon];

    // simulate rollout
    int t;
    for(t=0; t<timeHorizon; t++){
        int action;
        env.getFeatures(learner.features);
        learner.evaluate(env.validActions());
        if(randUniform() < epsilon){
            action = env.validActions()[randomN(env.validActions().size())];
        }
        else{
            action = sampleDist(learner.outputPolicy, numActions);
        }
        trajectory[t] = env;
        actions[t] = action;
        rewards[t] = env.makeAction(action);

        if(print){
            ofstream gameOut (gameOutFile, ios::app);
            gameOut << trajectory[t].toString() << '\n';
            gameOut << "Policy: ";
            for(int i=0; i<numActions; i++){
                gameOut << learner.outputPolicy[i] << ' ';
            }
            gameOut << '\n';
            gameOut << "Action: " << actions[t] << " Reward: " << rewards[t] << '\n';
        }
        if(env.endState) break;
    }
    int rolloutLength = t+1;

    // compute gradients
    double value = 0;
    double total_reward = 0;
    for(int t=rolloutLength-1; t>=0; t--){
        value *= discountFactor;
        value += rewards[t];
        total_reward += rewards[t];

        // baseline[t] = (1 - baselineRate) * baseline[t] + baselineRate * value;
        // unifBaseline = (1 - baselineRate) * unifBaseline + baselineRate * value;
        trajectory[t].getFeatures(learner.features);
        learner.accGrad(trajectory[t].validActions(), actions[t], value - unifBaseline);
    }
    return total_reward;
}

void PG::train(int batchSize, int numIter, double learnRate, double momentum){
    ofstream fout("score.out");
    double sum = 0;
    int evalPeriod = 1000;
    double evalSum = 0;
    unsigned start_time = time(0);
    string controlLog = "control.out";
    {
        ofstream controlOut ("control.out");
        controlOut.close();
    }
    for(int it=0; it<numIter; it++){
        learner.resetGrad();
        for(int i=0; i<batchSize; i++){
            double value = rollout();
            sum += value;
            if(it >= numIter/2){
                evalSum += value;
            }
        }
        learner.update(learnRate / batchSize, momentum);
        if(it % evalPeriod == 0){
            if(it > 0){
                fout << ',';
            }
            double avgScore = sum / batchSize / evalPeriod;
            fout << avgScore;
            {
                ofstream controlOut(controlLog, ios::app);
                controlOut << "Iteration " << it << " Time: " << (time(0) - start_time) << ' ' << avgScore << '\n';
                controlOut.close();
            }
            sum = 0;
        }
    }
    fout << '\n';
    {
        ofstream controlOut(controlLog, ios::app);
        controlOut << "Evaluation score: " << (evalSum / batchSize / (numIter/2)) << '\n';
        controlOut.close();
    }
}