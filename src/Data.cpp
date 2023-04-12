//
// Created by Aster on 02/07/2018.
//

#include "../include/Data.h"

vector<string> splitBySpace(string &sentence) {
    istringstream iss(sentence);
    return vector<string>{istream_iterator<string>(iss),
                          istream_iterator<string>{}};
}

void writeDataToCSV(vector<float> &results, Data &data,
                    const string &filename, bool train) {
    ofstream out(filename);
    if (out.is_open()) {
        out << "id,label";
        if (train) { out << ",real\n"; } else { out << "\n"; }
        int i = 0;
        for (auto each : results) {
            out << i << "," << each;
            if (train) {
                out << "," << data.readLabel(i) << "\n";
            } else {
                out << "\n";
            }
            i++;
        }
        out.close();
    } else {
        cout << "Write File failed" << endl;
    }
}

void writeDataToCSV(vector<int> &results, Data &data,
                    const string &filename, bool train) {
    ofstream out(filename);
    if (out.is_open()) {
        out << "id,label";
        if (train) { out << ",real\n"; } else { out << "\n"; }
        int i = 0;
        for (auto each : results) {
            out << i << "," << each;
            if (train) {
                out << "," << data.readLabel(i) << "\n";
            } else {
                out << "\n";
            }
            i++;
        }
        out.close();
    } else {
        cout << "Write File failed" << endl;
    }
}

Data::Data(bool isTrain, int size, int featuresSize) {
    this->featureSize = featuresSize;
    this->samplesSize = size;

    features.reserve(size);
    pSamples.reserve(size);

    if (isTrain) { target.reserve(size); }
    this->isTrain = isTrain;
}

void Data::read(const string &filename) {
    ifstream inputFile;
    inputFile.open(filename.c_str());

    if (!inputFile.is_open()) { cout << "Failed Open" << endl; }
    string str;

    for (int i = 0; i < this->samplesSize; i++)
    {
        (void)getline(inputFile, str);
        auto results = splitBySpace(str);

        if (this->featureSize != (results.size() - 1))
            cout << "FEAT_SIZE diff parsed feat: " << results.size() << endl;
        vector<float> sample(this->featureSize, 0);
        for (int i = 0; i < this->featureSize; i++)
        {
            sample[i] = stod(results[i]);
        }
        this->features.push_back(sample);
        if (this->isTrain)
        {
            /* The last member store the label [0, 1] */
            target.push_back(atoi(results[this->featureSize].c_str()));
        }
        pSamples.push_back(i);
    }
    inputFile.close();
    featuresVec.reserve(this->featureSize);
    for (int i = 0; i < featureSize; i++)
    {
        featuresVec.push_back(i);
    }
}

void Data::read(const string &filename, vector<int> &idx) {
    ifstream inputFile;
    inputFile.open(filename.c_str());

    if (!inputFile.is_open()) { cout << "Failed Open" << endl; }
    string str;

    for (int i = 0; i < this->samplesSize; i++)
    {
        (void)getline(inputFile, str);
        auto results = splitBySpace(str);

        vector<float> sample(this->featureSize, 0);
        for (int i = 0; i < this->featureSize; i++)
        {
            sample[i] = stod(results[idx[i]]);
        }
        this->features.push_back(sample);
        if (this->isTrain)
        {
            /* The last member store the label [0, 1] */
            target.push_back(atoi(results[this->featureSize].c_str()));
        }
        pSamples.push_back(i);
    }
    inputFile.close();
    featuresVec.reserve(this->featureSize);
    for (int i = 0; i < featureSize; i++)
    {
        featuresVec.push_back(i);
    }
}

float Data::readFeature(int sampleIndex, int featureIndex) {
    return features[sampleIndex][featureIndex];
}

int Data::readLabel(int sampleIndex) {
    return target[sampleIndex];
}

int Data::getSampleSize() {
    return (int) features.size();
}

int Data::getFeatureSize() {
    return featureSize;
}

vector<int> Data::generateSample(int &num) {
    if (num == -1) {
        return pSamples;
    } else {
        random_shuffle(pSamples.begin(), pSamples.end());
        return vector<int>(pSamples.begin(), pSamples.begin() + num);
    }
}

vector<int> Data::generateFeatures(function<int(int)> &func) {
    int m = func(this->getFeatureSize());
    random_shuffle(featuresVec.begin(), featuresVec.end());
    return vector<int>(featuresVec.begin(), featuresVec.begin() + m);
}

void Data::sortByFeature(vector<int> &pSamples, int featureIndex) {
    sort(pSamples.begin(), pSamples.end(), [&](int a, int b) {
        return this->readFeature(a, featureIndex) <
               this->readFeature(b, featureIndex);
    });
}
