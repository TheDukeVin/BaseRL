
/*

g++ -O2 -std=c++11 -pthread common.cpp MA_ttt/environment.cpp main.cpp MAPG.cpp -I "./LSTM" LSTM/node.cpp LSTM/model.cpp LSTM/PVUnit.cpp LSTM/layer.cpp LSTM/layers/lstmlayer.cpp LSTM/layers/policy.cpp LSTM/layers/conv.cpp LSTM/layers/pool.cpp LSTM/params.cpp && ./a.out && python3 comp.py

g++ -O2 -std=c++11 -pthread common.cpp MA_amazons/environment.cpp main.cpp MAPG.cpp -I "./LSTM" LSTM/node.cpp LSTM/model.cpp LSTM/PVUnit.cpp LSTM/layer.cpp LSTM/layers/lstmlayer.cpp LSTM/layers/policy.cpp LSTM/layers/conv.cpp LSTM/layers/pool.cpp LSTM/params.cpp && sbatch PG.slurm

-fsanitize=address -fsanitize=undefined -fno-sanitize-recover=all -fsanitize=float-divide-by-zero -fsanitize=float-cast-overflow -fno-sanitize=null -fno-sanitize=alignment

Sometimes FASRC pauses jobs for a while. Wait a couple days and try again.

*/

#include "MA_ttt/environment.h"
#include "lstm.h"


#ifndef MAPG_h
#define MAPG_h

class PGInstance{
public:
    Environment env;
    int action[numAgents];
    double reward[numAgents];
    double value[numAgents];
    double advantage[numAgents];
    double policy[numAgents][numActions];

    double importance_weight = -1;
};

class MAPG{
public:
    double alpha;
    double entropyConstant;

    const static double constexpr regRate = 0;
    // const static double constexpr GAEParam = 0.95;
    const static double constexpr GAEParam = 1;

    double valueNorm = 1;

    LSTM::PVUnit* structure;
    LSTM::PVUnit* net[numAgents][timeHorizon];

    string gameFile;
    string saveFile;
    string controlFile;
    string scoreFile;
    string firstFile;
    string hiddenFile;

    MAPG(LSTM::PVUnit* structure_, string gameFile_, string saveFile_, string controlFile_, string scoreFile_, string firstFile_, string hiddenFile_);

    long iterationCount;
    long start_time;

    double totalReward[numAgents];
    void rollout(bool train=true, bool print=false, bool outputHiddenVals=false); // edits totalReward array
    double multTotalReward[numAgents];
    void multRollout(int N);
    void accGrad(PGInstance instance, int index);

    // For evaluation:
    vector<LSTM::PVUnit*> netImages;
    vector<vector<double> > evaluation;
    int rolloutID;
    void train(int batchSize, int numIter, int evalPeriod, int savePeriod, int numEval, double alpha_, double entropyConstant_, int numThreads=1);

    void save();
    void load();
};

#endif