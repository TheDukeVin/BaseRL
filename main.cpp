

#include "PG.h"
// #include "DQN.h"

// ReplayBuffer buffer(100000);

int main(){
    unsigned start_time = time(0);

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

    // Snake environment

    // LSTM::Model structure = LSTM::Model(LSTM::Shape(boardx, boardy, 7));
    // // structure.addConv(LSTM::Shape(6, 6, 10), 3, 3);
    // // structure.addPool(LSTM::Shape(3, 3, 10));
    // // structure.addDense(40);
    // structure.addConv(LSTM::Shape(6, 6, 10), 5, 5);
    // structure.addConv(LSTM::Shape(6, 6, 10), 5, 5);
    // structure.addPool(LSTM::Shape(3, 3, 10));
    // structure.addDense(60);
    // structure.addOutput(numActions);
    // PG trainer(structure, "game.out");
    // trainer.train(1, 1000000, 0.001, -1);






    // For DQN


    // TTT environment

    // LSTM::Model structure = LSTM::Model(LSTM::Shape(numFeatures));
    // structure.addDense(40);
    // structure.addOutput(numActions);
    // DQN trainer(structure, "game.out", &buffer);
    // trainer.train(10, 1, 500000);

    // Snake environment

    // LSTM::Model structure = LSTM::Model(LSTM::Shape(boardx, boardy, 8));
    // structure.addConv(LSTM::Shape(6, 6, 7), 3, 3);
    // structure.addDense(100);
    // structure.addOutput(numActions);
    // DQN trainer(structure, "game.out", &buffer);
    // trainer.train(32, 15, 100000);

    LSTM::PVUnit structure;
    structure.commonBranch = new LSTM::Model(LSTM::Shape(boardx, boardy, 7));
    structure.commonBranch->addConv(LSTM::Shape(6, 6, 7), 3, 3);
    structure.initPV();
    structure.policyBranch->addDense(100);
    structure.policyBranch->addOutput(numActions);
    structure.valueBranch->addDense(50);
    structure.valueBranch->addOutput(1);
    PG_PV trainer(&structure, "game.out");
    trainer.train(1, 1000000);

    for(int i=0; i<5; i++){
        trainer.rollout(true);
    }
    cout << "Time: " << (time(0) - start_time) << '\n';
}