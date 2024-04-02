
/*
g++ -O2 -std=c++11 common.cpp snake/environment.cpp main.cpp DQN.cpp network_policy/policy.cpp -I "./LSTM" LSTM/node.cpp LSTM/model.cpp LSTM/PVUnit.cpp LSTM/layer.cpp LSTM/layers/lstmlayer.cpp LSTM/layers/policy.cpp LSTM/layers/conv.cpp LSTM/layers/pool.cpp LSTM/params.cpp && ./a.out

g++ -O2 -std=c++11 common.cpp snake/environment.cpp main.cpp DQN.cpp network_policy/policy.cpp -I "./LSTM" LSTM/node.cpp LSTM/model.cpp LSTM/PVUnit.cpp LSTM/layer.cpp LSTM/layers/lstmlayer.cpp LSTM/layers/policy.cpp LSTM/layers/conv.cpp LSTM/layers/pool.cpp LSTM/params.cpp && sbatch PG.slurm

rsync -r PG_test kevindu@login.rc.fas.harvard.edu:./MultiagentSnake --exclude .git/
rsync -r PG_test kevindu@login.rc.fas.harvard.edu:./MultiagentSnake/Dup --exclude .git/
*/

#include "snake/environment.h"
#include "../common.h"
#include "lstm.h"

#ifndef DQN_h
#define DQN_h

#define BufferSize 1000000

class ReplayInstance{
public:
    Environment currState;
    Environment nextState;
    int action;
    double reward;
    double value;

    ReplayInstance(){}
    ReplayInstance(Environment currState_, Environment nextState_, int action_, double reward_, double value_);
};

class ReplayBuffer{
private:
    ReplayInstance queue[BufferSize];
    int index;
    int fillSize;

public:
    int size;

    ReplayBuffer(int start_size);
    void updateSize(int new_size);
    void enqueue(ReplayInstance instance);

    ReplayInstance randomInstance();
};

class DQN{
public:
    const static double constexpr epsilon = 0.01;
    const static double constexpr eta = 0;
    const static double constexpr alpha = 0.001;
    const static double constexpr regRate = 0.00001;

    LSTM::Model net;
    LSTM::Data* netInput;
    LSTM::Data* netOutput;
    LSTM::Model structure;

    ReplayBuffer* buffer;

    string gameOutFile;

    DQN(LSTM::Model structure_, string gameOutFile_, ReplayBuffer* buffer_);

    double rolloutValue;
    double finalValue;
    void rollout(int batchSize, bool print=false);
    void updateInstance(ReplayInstance instance);
    void train(int batchSize, int numBatches, int numRollouts);
};

#endif