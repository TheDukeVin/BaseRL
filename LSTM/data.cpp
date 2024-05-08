#include "lstm.h"

using namespace LSTM;

Data::Data(int size_){
    size = size_;
    data = new double[size];
    gradient = new double[size];
    for(int i=0; i<size; i++){
        data[i] = gradient[i] = 0;
    }
}

Data::Data(int size_, double* data_, double* gradient_){
    size = size_;
    data = data_;
    gradient = gradient_;
}

void Data::resetGradient(){
    for(int i=0; i<size; i++){
        gradient[i] = 0;
    }
}

void Data::copy(Data* d){
    for(int i=0; i<size; i++){
        data[i] = d->data[i];
    }
}