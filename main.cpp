

// #include "PG.h"
// #include "DQN.h"
// #include "PPO.h"
#include "MAPG.h"

// #define NUM_THREADS 36



// ReplayBuffer buffer(100000);

// PPOStore dataset[NUM_THREADS];

// class Hyperparameter{
// public:
//     int numRollouts, batchSize, numEpochs, numIter, evalPeriod, savePeriod;
//     double alpha, GAEParam;

//     Hyperparameter(){}

//     Hyperparameter(int numRollouts_, int batchSize_, int numEpochs_, int numIter_, int evalPeriod_, int savePeriod_, double alpha_, double GAEParam_){
//         numRollouts = numRollouts_;
//         batchSize = batchSize_;
//         numEpochs = numEpochs_;
//         numIter = numIter_;
//         evalPeriod = evalPeriod_;
//         savePeriod = savePeriod_;
//         alpha = alpha_;
//         GAEParam = GAEParam_;
//     }

//     string toString(){
//         return "numRollouts: " + to_string(numRollouts) + " " +
//                "batchSize: " + to_string(batchSize) + " " +
//                "numEpochs: " + to_string(numEpochs) + " " +
//                "numIter: " + to_string(numIter) + " " +
//                "evalPeriod: " + to_string(evalPeriod) + " " +
//                "savePeriod: " + to_string(savePeriod) + " " +
//                "alpha: " + to_string(alpha) + " " +
//                "GAEParam: " + to_string(GAEParam) + " ";
//     }
// };

// Hyperparameter hps[NUM_THREADS];

// thread* threads[NUM_THREADS];

// double results[NUM_THREADS];

// void sweep(){
//     int count = 0;
//     // for(auto n : {1, 2, 4, 8}){
//     //     for(auto b : {64, 128, 256}){
//     //         for(auto l : {0.001, 0.0003, 0.0001}){
//     //             hps[count] = Hyperparameter(n, b, 1, 20000/n, 200/n, 200/n, l);
//     //             count ++;
//     //         }
//     //     }
//     // }
//     // for(auto n : {1, 2, 4, 8}){
//     //     for(auto l : {1e-03, 3e-04}){
//     //         for(auto e : {1, 2, 3}){
//     //             hps[count] = Hyperparameter(n, 256, e, 40000/n, 200/n, 200/n, l);
//     //             count ++;
//     //         }
//     //     }
//     // }
//     // for(auto n : {1, 2, 4, 8, 16}){
//     //     for(auto l : {1e-03, 5e-04}){
//     //         for(auto b : {256, 512}){
//     //             for(auto g : {1.0, 0.95}){
//     //                 hps[count] = Hyperparameter(n, b, 1, 100000/n, 400/n, 400/n, l, g);
//     //                 count ++;
//     //             }
//     //         }
//     //     }
//     // }

//     for(auto n : {1, 2, 4}){
//         for(auto l : {1e-03, 5e-04}){
//             for(auto b : {128, 512, NO_BATCH}){
//                 for(auto g : {1.0, 0.95}){
//                     hps[count] = Hyperparameter(n, b, 1, 100000/n, 400/n, 400/n, l, g);
//                     count ++;
//                 }
//             }
//         }
//     }
//     assert(count == NUM_THREADS);
// }

// void runThread(int i){
//     LSTM::PVUnit structure;
//     structure.commonBranch = new LSTM::Model(LSTM::Shape(boardx, boardy, 7));
//     structure.commonBranch->addConv(LSTM::Shape(10, 10, 10), 3, 3);
//     structure.commonBranch->addPool(LSTM::Shape(5, 5, 10));
//     structure.commonBranch->addDense(250);
//     structure.initPV();
//     structure.policyBranch->addOutput(numActions);
//     structure.valueBranch->addOutput(1);
//     string s = "PPOsweep/session" + to_string(i);
//     PPO trainer(&structure, &dataset[i], s + "game.out", s + "save.out", s + "control.out", s + "score.out");
//     results[i] = trainer.train(hps[i].numRollouts, hps[i].batchSize, hps[i].numEpochs, hps[i].numIter, hps[i].evalPeriod, hps[i].savePeriod, hps[i].alpha, hps[i].GAEParam, 0);
// }

// void runSweep(){
//     sweep();
//     for(int i=0; i<NUM_THREADS; i++){
//         threads[i] = new thread(runThread, i);
//     }
//     for(int i=0; i<NUM_THREADS; i++){
//         threads[i]->join();
//     }
//     for(int i=0; i<NUM_THREADS; i++){
//         cout << results[i] << " " << hps[i].toString() << '\n';
//     }
// }

int main(){
    unsigned start_time = time(0);

    // double dist[4] = {0.1, 0.2, 0.3, 0.4};
    // int samples[10];
    // sampleMult(dist, 4, samples, 10);
    // for(int i=0; i<10; i++){
    //     cout << samples[i] << ' ';
    // }
    // cout << '\n';

    // Environment env;
    // env.grid[6][0] = 3;
    // env.grid[6][1] = 2;
    // env.grid[6][2] = 2;
    // env.grid[5][2] = 1;
    // env.grid[4][2] = 1;
    // env.grid[3][2] = 1;
    // env.grid[3][1] = 0;
    // env.grid[3][0] = 0;
    // env.grid[2][0] = 1;
    // env.snake.tail = Pos(2, 0);
    // cout << env.toString() << '\n';
    // cout << env.potential() << '\n';

    // PG algorithms

    // Collect gem environment

    // LSTM::Model structure = LSTM::Model(LSTM::Shape(numFeatures));
    // structure.addDense(20);
    // structure.addOutput(numActions);
    // PG trainer(structure);
    // trainer.train(100, 30000, 0.01);

    // TTT environment

    // LSTM::PVUnit structure;
    // structure.commonBranch = new LSTM::Model(LSTM::Shape(numFeatures));
    // structure.commonBranch->addDense(20);
    // structure.initPV();
    // structure.policyBranch->addOutput(numActions);
    // structure.valueBranch->addOutput(1);
    // PG_LSTM trainer(&structure, "game.out", "save.out", "control.out", "score.out");
    // trainer.train(20, 200000, 10000, 10000, 2e-03);

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

    // SNAKE

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

    // runSweep();

    // LSTM::PVUnit structure;
    // structure.commonBranch = new LSTM::Model(LSTM::Shape(boardx, boardy, 7));
    // structure.commonBranch->addConv(LSTM::Shape(10, 10, 10), 3, 3);
    // structure.commonBranch->addPool(LSTM::Shape(5, 5, 10));
    // structure.commonBranch->addDense(250);
    // structure.initPV();
    // structure.policyBranch->addOutput(numActions);
    // structure.valueBranch->addOutput(1);
    // int i = 0;
    // string s = "PPOsweep/session" + to_string(i);
    // PPO trainer(&structure, &dataset[i], s + "game.out", s + "save.out", s + "control.out", s + "score.out");
    // results[i] = trainer.train(2, 512, 1, 250000, 200, 200, 0.001, 0.95, 0);

    // LSTM::PVUnit structure;
    // structure.commonBranch = new LSTM::Model(LSTM::Shape(boardx, boardy, 7));
    // structure.commonBranch->addConv(LSTM::Shape(10, 10, 10), 3, 3);
    // structure.commonBranch->addPool(LSTM::Shape(5, 5, 10));
    // structure.initPV();
    // structure.policyBranch->addDense(150);
    // structure.policyBranch->addOutput(numActions);
    // structure.valueBranch->addDense(100);
    // structure.valueBranch->addOutput(1);
    // string s = "PPOsweep/session3";
    // PPO trainer(&structure, &dataset[3], "game.out", s + "save.out", s + "control.out", s + "score.out");
    // trainer.load();
    
    // LSTM::PVUnit structure;
    // structure.commonBranch = new LSTM::Model(LSTM::Shape(boardx, boardy, 7));
    // structure.commonBranch->addConv(LSTM::Shape(10, 10, 10), 3, 3);
    // structure.commonBranch->addPool(LSTM::Shape(5, 5, 10));
    // structure.commonBranch->addDense(250);
    // structure.initPV();
    // structure.policyBranch->addOutput(numActions);
    // structure.valueBranch->addOutput(1);
    // string s = "PPOsweep/session0";
    // PPO trainer(&structure, &dataset[0], s + "game.out", s + "save.out", s + "control.out", s + "score.out");
    // // trainer.train(1, 512, -1, 10000, 100, 100, 5e-04, 0.95, 0.3, true);
    // trainer.train(1, 512, 1, 10000, 100, 100, 1e-03, 0.95);






    // LSTM PG

    // Repeat environment

    // LSTM::PVUnit structure;
    // structure.commonBranch = new LSTM::Model(LSTM::Shape(numFeatures));
    // structure.commonBranch->addLSTM(8);
    // structure.initPV();
    // structure.policyBranch->addOutput(numActions);
    // structure.valueBranch->addOutput(1);
    // PG_LSTM trainer(&structure, "game.out", "save.out", "control.out", "score.out");
    // trainer.train(10, 100000, 10000, 10000, 0.001);



    // Token environment

    // LSTM::PVUnit structure;
    // structure.commonBranch = new LSTM::Model(LSTM::Shape(numFeatures));
    // structure.commonBranch->addLSTM(8);
    // structure.initPV();
    // structure.policyBranch->addOutput(numActions);
    // structure.valueBranch->addOutput(1);
    // PG_LSTM trainer(&structure, "game.out", "save.out", "control.out", "score.out");
    // trainer.train(1, 50000, 1000, 1000, 0.001);


    // Snake environment

    // LSTM::PVUnit structure;
    // structure.commonBranch = new LSTM::Model(LSTM::Shape(boardx, boardy, 7));
    // structure.commonBranch->addConv(LSTM::Shape(10, 10, 10), 3, 3);
    // structure.commonBranch->addPool(LSTM::Shape(5, 5, 10));
    // structure.commonBranch->addLSTM(100);
    // structure.initPV();
    // structure.policyBranch->addOutput(numActions);
    // structure.valueBranch->addOutput(1);
    // PG_LSTM trainer(&structure, "game.out", "save.out", "control.out", "score.out");
    // trainer.train(5, 5000, 100, 100, 5e-04);


    // Search

    // LSTM::PVUnit structure;
    // structure.commonBranch = new LSTM::Model(LSTM::Shape(boardx, boardy, 2));
    // structure.commonBranch->addConv(LSTM::Shape(10, 10, 3), 3, 3);
    // structure.commonBranch->addPool(LSTM::Shape(5, 5, 3));
    // structure.commonBranch->addLSTM(50);
    // structure.initPV();
    // structure.policyBranch->addOutput(numActions);
    // structure.valueBranch->addOutput(1);
    // PG_LSTM trainer(&structure, "game.out", "save.out", "control.out", "score.out");
    // trainer.train(5, 100000, 2000, 2000, 0.002);
    // trainer.load();


    // Radar

    // LSTM::PVUnit structure;
    // structure.commonBranch = new LSTM::Model(LSTM::Shape(boardx, boardy, 2));
    // structure.commonBranch->addConv(LSTM::Shape(10, 10, 3), 3, 3);
    // structure.commonBranch->addPool(LSTM::Shape(5, 5, 3));
    // structure.commonBranch->addLSTM(50);
    // structure.initPV();
    // structure.policyBranch->addOutput(numActions);
    // structure.valueBranch->addOutput(1);
    // PG_LSTM trainer(&structure, "game.out", "save.out", "control.out", "score.out");
    // trainer.train(5, 50000, 2000, 2000, 0.002);


    // Poker

    // LSTM::PVUnit structure;
    // structure.commonBranch = new LSTM::Model(numFeatures);
    // structure.commonBranch->addDense(100);
    // structure.initPV();
    // structure.policyBranch->addOutput(numActions);
    // structure.valueBranch->addOutput(1);
    // PG_LSTM trainer(&structure, "game.out", "save.out", "control.out", "score.out");
    // trainer.train(20, 50000, 500, 500, 2e-03);

    // LSTM::PVUnit structure;
    // structure.commonBranch = new LSTM::Model(LSTM::Shape(boardx, boardy, 2));
    // structure.commonBranch->addConv(LSTM::Shape(10, 10, 3), 3, 3);
    // structure.commonBranch->addPool(LSTM::Shape(5, 5, 3));
    // structure.commonBranch->addDense(50);
    // structure.initPV();
    // structure.policyBranch->addOutput(numActions);
    // structure.valueBranch->addOutput(1);
    // PG_LSTM trainer(&structure, "game.out", "save.out", "control.out", "score.out");
    // trainer.train(5, 100000, 2000, 2000, 0.002);






    // MA PG

    // Matrix game

    // LSTM::PVUnit structure;
    // structure.commonBranch = new LSTM::Model(2);
    // structure.commonBranch->addDense(3);
    // structure.initPV();
    // structure.policyBranch->addOutput(numActions);
    // structure.valueBranch->addOutput(1);

    // MAPG trainer(&structure, "game.out", "save.out", "control.out", "score.out", "first.out");
    // trainer.train(50, 5000, 500, 500, 1e-03);

    // MA ttt

    LSTM::PVUnit structure;
    structure.commonBranch = new LSTM::Model(numFeatures);
    // structure.commonBranch->addDense(20);
    structure.commonBranch->addLSTM(20);
    structure.initPV();
    structure.policyBranch->addOutput(numActions);
    structure.valueBranch->addOutput(1);

    MAPG trainer(&structure, "game.out", "save.out", "control.out", "score.out", "first.out", "hidden.out");
    // trainer.train(50, 50000, 5000, 5000, 1000, 1e-03, 1e-01, 2);
    trainer.load();
    for(int ag=0; ag<numAgents; ag++){
        for(int t=0; t<timeHorizon; t++){
            trainer.net[ag][t]->copyParams(trainer.structure);
        }
    }
    {
        ofstream fout("hidden.out");
    }
    for(int i=0; i<10000; i++){
        trainer.rollout(false, false, true);
    }
    
    // Amazons

    // LSTM::PVUnit structure;
    // structure.commonBranch = new LSTM::Model(LSTM::Shape(boardHeight, boardWidth, 2*numAmazons + 3));
    // structure.commonBranch->addConv(LSTM::Shape(6, 6, 10), 3, 3);
    // structure.commonBranch->addConv(LSTM::Shape(4, 4, 15), 3, 3);
    // structure.commonBranch->addLSTM(100);
    // structure.initPV();
    // structure.policyBranch->addOutput(numActions);
    // structure.valueBranch->addOutput(1);

    // MAPG trainer(&structure, "game.out", "save.out", "control.out", "score.out", "first.out");
    // // trainer.train(50, 20000, 1000, 1000, 200, 1e-03, 10);
    // trainer.train(32, 200000, 10000, 10000, 1000, 1e-03, 16);

    // Liars dice

    // LSTM::PVUnit structure;
    // structure.commonBranch = new LSTM::Model(numFeatures);
    // structure.commonBranch->addLSTM(40);
    // structure.commonBranch->addLSTM(20);
    // structure.initPV();
    // structure.policyBranch->addOutput(numActions);
    // structure.valueBranch->addOutput(1);

    // MAPG trainer(&structure, "game.out", "save.out", "control.out", "score.out", "first.out");
    // trainer.train(64, 150000, 10000, 10000, 1000, 1e-03, 2);


    {
        ofstream codeOut("code.out");
    }
    for(int ag=0; ag<numAgents; ag++){
        for(int t=0; t<timeHorizon; t++){
            trainer.net[ag][t]->copyParams(trainer.structure);
        }
    }
    for(int i=0; i<5; i++){
        trainer.rollout(true);
    }
    cout << "Time: " << (time(0) - start_time) << '\n';
}
