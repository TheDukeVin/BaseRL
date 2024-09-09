
/*

g++ -O2 -std=c++11 common.cpp MA_ttt/environment.cpp main.cpp MAPG.cpp -I "./LSTM" LSTM/node.cpp LSTM/model.cpp LSTM/PVUnit.cpp LSTM/layer.cpp LSTM/layers/lstmlayer.cpp LSTM/layers/policy.cpp LSTM/layers/conv.cpp LSTM/layers/pool.cpp LSTM/params.cpp && ./a.out && python3 comp.py

-fsanitize=address -fsanitize=undefined -fno-sanitize-recover=all -fsanitize=float-divide-by-zero -fsanitize=float-cast-overflow -fno-sanitize=null -fno-sanitize=alignment


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
    const static double constexpr regRate = 0;
    const static double constexpr entropyConstant = 1e-01;
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

    MAPG(LSTM::PVUnit* structure_, string gameFile_, string saveFile_, string controlFile_, string scoreFile_, string firstFile_);

    long iterationCount;
    long start_time;

    double totalReward[numAgents];
    void rollout(bool print=false); // edits totalReward array
    void accGrad(PGInstance instance, int index);

    // For evaluation:
    vector<LSTM::PVUnit*> netImages;
    int rolloutID;
    void train(int batchSize, int numIter, int evalPeriod, int savePeriod, int numEval, double alpha_);

    void save();
    void load();
};

#endif