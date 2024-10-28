
/*
g++ -O2 -std=c++11 -pthread -fsanitize=address -fsanitize=undefined -fno-sanitize-recover=all -fsanitize=float-divide-by-zero -fsanitize=float-cast-overflow -fno-sanitize=null -fno-sanitize=alignment common.cpp MA_ttt/environment.cpp autoencoder.cpp MAPG.cpp -I "./LSTM" LSTM/node.cpp LSTM/model.cpp LSTM/PVUnit.cpp LSTM/layer.cpp LSTM/layers/lstmlayer.cpp LSTM/layers/policy.cpp LSTM/layers/conv.cpp LSTM/layers/pool.cpp LSTM/params.cpp && ./a.out

-fsanitize=address -fsanitize=undefined -fno-sanitize-recover=all -fsanitize=float-divide-by-zero -fsanitize=float-cast-overflow -fno-sanitize=null -fno-sanitize=alignment
*/

#include "lstm.h"
#include "/Users/kevindu/Desktop/Coding/Tests:experiments/PG_test/common.h"

class AutoEncoder{
public:
    vector<vector<double> > data;
    LSTM::Model* encoderStructure;

    LSTM::Model* encoder;
    LSTM::Data* input;
    LSTM::Data* output;

    vector<double> errorHist;

    AutoEncoder(vector<vector<double> > data_, LSTM::Model* encoderStructure_){
        input = new LSTM::Data(data_[0].size());
        output = new LSTM::Data(data_[0].size());
        data = data_;
        encoderStructure = new LSTM::Model(encoderStructure_, NULL, input, output);

        encoderStructure->randomize(0.1);
        encoderStructure->resetGradient();

        encoder = new LSTM::Model(encoderStructure, NULL, input, output);
        encoder->copyParams(encoderStructure);
    }

    double accGrad(){ // returns reconstruction error
        int index = randomN(data.size());
        for(int i=0; i<data[index].size(); i++){
            input->data[i] = data[index][i];
        }
        encoder->forwardPass();
        double sum = 0;
        for(int i=0; i<data[index].size(); i++){
            output->gradient[i] = output->data[i] - data[index][i];
            sum += pow(output->gradient[i], 2);
        }
        encoder->backwardPass();
        encoderStructure->accumulateGradient(encoder);
        return sum;
    }

    void train(int numIter, int batchSize, double alpha){
        for(int t=0; t<numIter; t++){
            double errorSum = 0;
            for(int i=0; i<batchSize; i++){
                errorSum += accGrad();
            }
            errorHist.push_back(errorSum/batchSize);
            encoderStructure->updateParams(alpha);
            encoder->copyParams(encoderStructure);
        }
    }

    void outputTransformedValues(string fileName, int layerIndex){
        ofstream fout(fileName);
        for(int i=0; i<data.size(); i++){
            for(int j=0; j<data[0].size(); j++){
                input->data[j] = data[i][j];
            }
            encoder->forwardPass();
            for(int j=0; j<encoder->activations[layerIndex]->size; j++){
                fout << encoder->activations[layerIndex]->data[j] << ' ';
            }
            fout << '\n';
        }
    }
};

void quick_test(){
    vector<vector<double> > data;
    data.push_back(vector<double>{0, 1});
    data.push_back(vector<double>{1, 2});
    data.push_back(vector<double>{5, 1});
    data.push_back(vector<double>{3, 8});

    LSTM::Model* structure = new LSTM::Model(LSTM::Shape(2));
    structure->addDense(1, "identity");
    structure->addOutput(2);

    AutoEncoder ae(data, structure);

    ae.train(1000, 1, 0.01);

    ofstream fout ("autoencoder_test/auto_error.out");

    for(auto a : ae.errorHist){
        fout << a << ' ';
    }
}

vector<vector<double> > readData(string fileName, int N, int d){
    vector<vector<double> > data;
    ifstream fin(fileName);
    for(int i=0; i<N; i++){
        vector<double> row;
        for(int j=0; j<d; j++){
            double x; fin >> x;
            row.push_back(x);
        }
        data.push_back(row);
    }
    return data;
}

vector<vector<double> > readLines(string fileName, int d){
    vector<vector<double> > data;
    ifstream fin(fileName);
    string s;
    while(getline(fin, s)){
        vector<double> row;
        stringstream sin(s);
        for(int j=0; j<d; j++){
            double x; sin >> x;
            row.push_back(x);
        }
        data.push_back(row);
    }
    return data;
}

int main(){
    // vector<vector<double> > data = readData("autoencoder_test/data.out", 100, 3);
    vector<vector<double> > data = readLines("autoencoder_test/data.out", 20);
    cout << data.size() << ' ' << data[0].size() << '\n';
    LSTM::Model* structure = new LSTM::Model(LSTM::Shape(20));
    structure->addDense(15, "leakyRelu");
    structure->addDense(2, "identity");
    structure->addDense(15, "leakyRelu");
    structure->addOutput(20);
    AutoEncoder ae(data, structure);

    ae.train(10000, 20, 0.01);

    ofstream fout ("autoencoder_test/auto_error.out");

    for(auto a : ae.errorHist){
        fout << a << ' ';
    }

    ae.outputTransformedValues("autoencoder_test/transform.out", 2);
}