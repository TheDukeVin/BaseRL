
/*
g++ -O2 -std=c++11 common.cpp snake/environment.cpp main.cpp PG.cpp PG_PV.cpp network_policy/policy.cpp -I "./LSTM" LSTM/node.cpp LSTM/model.cpp LSTM/PVUnit.cpp LSTM/layer.cpp LSTM/layers/lstmlayer.cpp LSTM/layers/policy.cpp LSTM/layers/conv.cpp LSTM/layers/pool.cpp LSTM/params.cpp && ./a.out

g++ -O2 -std=c++11 common.cpp snake/environment.cpp main.cpp PG.cpp PG_PV.cpp network_policy/policy.cpp -I "./LSTM" LSTM/node.cpp LSTM/model.cpp LSTM/PVUnit.cpp LSTM/layer.cpp LSTM/layers/lstmlayer.cpp LSTM/layers/policy.cpp LSTM/layers/conv.cpp LSTM/layers/pool.cpp LSTM/params.cpp && sbatch PG.slurm

rsync -r PG_test kevindu@login.rc.fas.harvard.edu:./MultiagentSnake --exclude .git/
rsync -r PG_test kevindu@login.rc.fas.harvard.edu:./MultiagentSnake/Dup --exclude .git/
*/

#include "snake/environment.h"
#include "network_policy/policy.h"

#ifndef PG_h
#define PG_h


class PG{
public:
    const static double constexpr epsilon = 0.01;

    double baseline[timeHorizon];
    double unifBaseline = 0;
    const static double constexpr baselineRate = 0.0001;

    Policy learner;
    string gameOutFile;

    PG(LSTM::Model structure_, string gameOutFile_);
    double rollout(bool print=false);
    void train(int batchSize, int numIter, double learnRate, double momentum);
};

class PGInstance{
public:
    Environment env;
    int action;
    double reward;
    double value;
    double policy[numActions];
};

class PG_PV{
public:
    const static double constexpr epsilon = 0;
    const static double constexpr alpha = 0.001;
    const static double constexpr regRate = 0.0001;
    const static double constexpr entropyConstant = 0.01;

    double valueFirstMoment = 0;
    double valueSecondMoment = 0;
    double valueNorm = 1;
    long valueUpdateCount = 0;
    const static double constexpr valueUpdateRate = 0.001 / timeHorizon;
    const static double constexpr valueNormConstant = 2;

    LSTM::PVUnit* structure;
    LSTM::PVUnit* net;

    string gameOutFile;

    PG_PV(LSTM::PVUnit* structure_, string gameOutFile_);

    double rolloutValue;
    double finalValue;
    double rolloutTime;
    void rollout(bool print=false);
    void accGrad(PGInstance instance);
    void train(int batchSize, int numIter);
};

#endif