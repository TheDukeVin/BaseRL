

// #include "PG.h"
// #include "DQN.h"
#include "PPO.h"

// ReplayBuffer buffer(100000);

PPOStore dataset;

int main(){
    unsigned start_time = time(0);

    // PG algorithms

    // Collect gem environment

    // LSTM::Model structure = LSTM::Model(LSTM::Shape(numFeatures));
    // structure.addDense(20);
    // structure.addOutput(numActions);
    // PG trainer(structure);
    // trainer.train(100, 30000, 0.01);

    // TTT environment

    // LSTM::Model structure = LSTM::Model(LSTM::Shape(numFeatures));
    // structure.addDense(20);
    // structure.addOutput(numActions);
    // PG trainer(structure, "game.out");
    // trainer.train(10, 300000, 0.001, -1);

    // Token environment

    // LSTM::PVUnit structure;
    // structure.commonBranch = new LSTM::Model(LSTM::Shape(numFeatures));
    // structure.commonBranch->addDense(6);
    // structure.initPV();
    // structure.policyBranch->addOutput(4);
    // structure.valueBranch->addOutput(1);
    // PG_PV trainer(&structure, "game.out");
    // trainer.train(1, 50000);

    // Snake environment

    // LSTM::PVUnit structure;
    // structure.commonBranch = new LSTM::Model(LSTM::Shape(boardx, boardy, 7));
    // structure.commonBranch->addConv(LSTM::Shape(6, 6, 7), 3, 3);
    // structure.initPV();
    // structure.policyBranch->addDense(100);
    // structure.policyBranch->addOutput(numActions);
    // structure.valueBranch->addDense(50);
    // structure.valueBranch->addOutput(1);
    // PG_PV trainer(&structure, "game.out", "save.out");
    // trainer.train(1, 1000000);

    




    // DQN algorithms

    // Snake

    // LSTM::Model structure = LSTM::Model(LSTM::Shape(boardx, boardy, 8));
    // structure.addConv(LSTM::Shape(6, 6, 7), 3, 3);
    // structure.addDense(100);
    // structure.addOutput(numActions);
    // DQN trainer(structure, "game.out", &buffer);
    // trainer.train(32, 15, 100000);




    // PPO algorithms

    // LSTM::PVUnit structure;
    // structure.commonBranch = new LSTM::Model(LSTM::Shape(boardx, boardy, 7));
    // structure.commonBranch->addConv(LSTM::Shape(6, 6, 7), 3, 3);
    // structure.initPV();
    // structure.policyBranch->addDense(100);
    // structure.policyBranch->addOutput(numActions);
    // structure.valueBranch->addDense(50);
    // structure.valueBranch->addOutput(1);
    // PPO trainer(&structure, &dataset, "game.out", "save.out", "control.out", "score.out");
    // trainer.train(10, 64, 1, 4000);

    LSTM::PVUnit structure;
    structure.commonBranch = new LSTM::Model(LSTM::Shape(boardx, boardy, 7));
    structure.commonBranch->addConv(LSTM::Shape(10, 10, 10), 3, 3);
    structure.commonBranch->addPool(LSTM::Shape(5, 5, 10));
    structure.initPV();
    structure.policyBranch->addDense(150);
    structure.policyBranch->addOutput(numActions);
    structure.valueBranch->addDense(100);
    structure.valueBranch->addOutput(1);
    PPO trainer(&structure, &dataset, "game.out", "save.out", "control.out", "score.out");
    trainer.train(2, 64, 1, 100);

    for(int i=0; i<5; i++){
        trainer.rollout(true);
    }
    cout << "Time: " << (time(0) - start_time) << '\n';
}