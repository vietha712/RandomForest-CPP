//
// Created by Aster on 23/06/2018.
//

#ifndef RANDOMFOREST_DATA_H
#define RANDOMFOREST_DATA_H

#include <iostream>
#include <vector>
#include <map>
#include <fstream>
#include <string>
#include <sstream>
#include <iostream>
#include <random>
#include <algorithm>
#include <ctime>
#include <functional>
#include <iterator>

using namespace std;

vector<string> splitBySpace(string &sentence);

class Data {
private:
    vector<vector<float>> features;
    vector<int> target;
    int featureSize = 0;
    int samplesSize = 0;
    bool isTrain;
    vector<int> featuresVec;
    vector<int> pSamples;

public:
    Data(bool isTrain, int size, int featuresSize);

    void read(const string &filename);

    void read(const string &filename, vector<int> &idx);

    float readFeature(int sampleIndex, int featureIndex);

    int readLabel(int sampleIndex);

    int getSampleSize();

    int getFeatureSize();

    vector<int> generateSample(int &num);

    vector<int> generateFeatures(function<int(int)> &func);

    void sortByFeature(vector<int> &pSamples, int featureIndex);
};

void writeDataToCSV(vector<float> &results,
                    Data &data,
                    const string &filename,
                    bool train);

void writeDataToCSV(vector<int> &results,
                    Data &data,
                    const string &filename,
                    bool train);

#endif //RANDOMFOREST_DATA_H
