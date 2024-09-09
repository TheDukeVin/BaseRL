
/*
g++ -pthread -O2 -std=c++11 common.cpp snake/environment.cpp main.cpp PPO.cpp symunit.cpp network_policy/policy.cpp -I "./LSTM" LSTM/node.cpp LSTM/model.cpp LSTM/PVUnit.cpp LSTM/layer.cpp LSTM/layers/lstmlayer.cpp LSTM/layers/policy.cpp LSTM/layers/conv.cpp LSTM/layers/pool.cpp LSTM/params.cpp && ./a.out

g++ -pthread -O2 -std=c++11 common.cpp snake/environment.cpp main.cpp PPO.cpp symunit.cpp network_policy/policy.cpp -I "./LSTM" LSTM/node.cpp LSTM/model.cpp LSTM/PVUnit.cpp LSTM/layer.cpp LSTM/layers/lstmlayer.cpp LSTM/layers/policy.cpp LSTM/layers/conv.cpp LSTM/layers/pool.cpp LSTM/params.cpp && sbatch PG.slurm

-fsanitize=address -fsanitize=undefined -fno-sanitize-recover=all -fsanitize=float-divide-by-zero -fsanitize=float-cast-overflow -fno-sanitize=null -fno-sanitize=alignment


rsync -r PG_test kevindu@login.rc.fas.harvard.edu:./MultiagentSnake --exclude .git/
rsync -r PG_test kevindu@login.rc.fas.harvard.edu:./MultiagentSnake/Dup --exclude .git/


Generalized Advantage Estimation:
https://arxiv.org/pdf/1506.02438.pdf
PPO paper:
https://arxiv.org/pdf/1707.06347.pdf
Reward shaping:
https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf

*/

#include "snake/environment.h"
#include "common.h"
#include "lstm.h"
#include "PG.h"

#ifndef PPO_h
#define PPO_h

#define BufferSize 6000

#define NO_BATCH 0

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
    double alpha;
    double GAEParam;
    double importance_temp;

    const static double constexpr regRate = 0;
    const static double constexpr entropyConstant = 0.01;
    const static double constexpr clipRange = 0.1;
    const static double constexpr valueNormOverride = 5; // use if you have a good idea for the value norm.
    // const static double constexpr GAEParam = 1;

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
    double valueNorm = valueNormOverride;
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
    int gameCount = 0;
    double sumScore = 0;
    double sumSize = 0;
    double lossCount = 0;
    double winCount = 0;
    double winTime = 0;

    // generates new rollouts. logs performance in controlFile.
    // ensures dataset size is greater than batchSize.
    void generateDataset(int numRollouts, int batchSize);

    void accGrad(PGInstance instance);
    void trainEpoch(int batchSize, bool pool);

    // train network. returns evaluation score;
    // pool=true: pool training data in experience pool rather than queue.
    double train(int numRollouts, int batchSize, int numEpochs, int numIter, int evalPeriod, int savePeriod, double alpha_, double GAEParam_, double importance_temp_, bool pool = false);

    void save();
    void load();
};

#endif