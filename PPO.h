
/*
g++ -O2 -std=c++11 common.cpp snake/environment.cpp main.cpp PPO.cpp symunit.cpp network_policy/policy.cpp -I "./LSTM" LSTM/node.cpp LSTM/model.cpp LSTM/PVUnit.cpp LSTM/layer.cpp LSTM/layers/lstmlayer.cpp LSTM/layers/policy.cpp LSTM/layers/conv.cpp LSTM/layers/pool.cpp LSTM/params.cpp && ./a.out

g++ -O2 -std=c++11 common.cpp snake/environment.cpp main.cpp PPO.cpp symunit.cpp network_policy/policy.cpp -I "./LSTM" LSTM/node.cpp LSTM/model.cpp LSTM/PVUnit.cpp LSTM/layer.cpp LSTM/layers/lstmlayer.cpp LSTM/layers/policy.cpp LSTM/layers/conv.cpp LSTM/layers/pool.cpp LSTM/params.cpp && sbatch PG.slurm

-fsanitize=address -fsanitize=undefined -fno-sanitize-recover=all -fsanitize=float-divide-by-zero -fsanitize=float-cast-overflow -fno-sanitize=null -fno-sanitize=alignment


rsync -r PG_test kevindu@login.rc.fas.harvard.edu:./MultiagentSnake --exclude .git/
rsync -r PG_test kevindu@login.rc.fas.harvard.edu:./MultiagentSnake/Dup --exclude .git/


Generalized Advantage Estimation:
https://arxiv.org/pdf/1506.02438.pdf
PPO paper:
https://arxiv.org/pdf/1707.06347.pdf

*/

#include "snake/environment.h"
#include "../common.h"
#include "lstm.h"

#ifndef PPO_h
#define PPO_h

#define BufferSize 5000

class PGInstance{
public:
    Environment env;
    int action;
    double reward;
    double value;
    double advantage;
    double policy[numActions];
};

class PPOStore{
public:
    int index = 0;
    PGInstance queue[BufferSize];

    void empty();
    void enqueue(PGInstance instance);
    void shuffleQueue();
    int getSize();
};

class SymUnit{
private:

    int symID;

public:

    LSTM::PVUnit* structure;
    LSTM::PVUnit* net;
    LSTM::Data* valueOutput;
    LSTM::Data* policyOutput;

    SymUnit(){}
    SymUnit(LSTM::PVUnit* structure_);
    void forwardPass(Environment env, int symID_);
    void backwardPass();
    void update(double alpha, double regRate);
};

class PPO{
public:
    //Geometric annealing
    const static double constexpr startingAlpha = 1e-03;
    const static double constexpr terminalAlpha = 1e-03;
    double alpha;

    const static double constexpr regRate = 0;
    const static double constexpr entropyConstant = 0.01;
    const static double constexpr clipRange = 0.1;
    const static double constexpr valueNormOverride = 5; // use if you have a good idea for the value norm.
    const static double constexpr GAEParam = 0.95;

    const static double constexpr valueUpdateRate = 0.1 / BufferSize;
    const static double constexpr valueNormConstant = 2;

    // LSTM::PVUnit* structure;
    // LSTM::PVUnit* net;

    SymUnit symNet;

    PPOStore* dataset;

    string gameFile;
    string saveFile;
    string controlFile;
    string scoreFile;

    double valueFirstMoment = 0;
    double valueSecondMoment = 0;
    double valueNorm = 0;
    long valueUpdateCount = 0;
    long iterationCount = 0;
    long start_time;

    PPO(LSTM::PVUnit* structure_, PPOStore* dataset_, string gameFile_, string saveFile_, string controlFile_, string scoreFile_);

    // tracks valueNorm
    double rolloutValue;
    double finalValue;
    int rolloutTime;
    void rollout(bool print=false);

    // Evaluation benchmarks
    double sumScore = 0;
    double lossCount = 0;
    double winCount = 0;
    double winTime = 0;

    // generates new rollouts. logs performance in controlFile.
    // ensures dataset size is greater than batchSize.
    void generateDataset(int numRollouts, int batchSize);

    void accGrad(PGInstance instance);
    void trainEpoch(int batchSize);
    void train(int numRollouts, int batchSize, int numEpochs, int numIter);

    void save();
    void load();
};

#endif