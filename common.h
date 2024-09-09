
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iterator>
#include <random>
#include <vector>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <ctime>
#include <math.h>
#include <utility>
#include <cassert>

#ifndef common_h
#define common_h

using namespace std;

static std::random_device dev;
static std::mt19937 rng(dev());

int randomN(int N);
double randUniform();
void computeSoftmaxPolicy(double* logits, int size, vector<int> validActions, double* policy); // -1 means invalid action.
int sampleDist(double* dist, int N); // -1 represents an invalid value.
void sampleMult(double* dist, int N, int* samples, int numSamples);

#endif