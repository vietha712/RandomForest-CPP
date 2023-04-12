//
// Created by Aster on 25/06/2018.
//

#ifndef RANDOMFOREST_DECISIONTREE_H
#define RANDOMFOREST_DECISIONTREE_H

#include "Data.h"
#include <memory>
#include <functional>
#include <set>
#include <utility>
#include <cmath>

using namespace std;

float computeTargetProb(int* pSamples, int samplesSize, Data &Data);

float computeEntropy(vector<int> &samples, Data &Data);

float computeGini(int &, int &);

//float computeInformationGain(vector<int> &samples,
//                              vector<int> &pSamplesLeft,
//                              vector<int> &pSamplesRight,
//                              Data &Data);

float computeGiniIndex(int &, int &, int &, int &);

int _sqrt(int num);

int _log2(int num);

int _none(int num);

struct Node {
    int featureIndex;
    Node* pLeft;
    Node* pRight;
    float threshold;
    bool isLeaf;
    int depth;
    float prob;
    Node() {
        pLeft = nullptr;
        pRight = nullptr;
        isLeaf = false;
    }
};

class DecisionTree {
private:
    int featureNum;
    int maxDepth;
    int minSamplesSplit;
    int minSamplesLeaf;
    int sampleNum;
    function<float(int&, int&, int&, int&)> criterionFunc;
    function<int(int)> maxFeatureFunc;
    Node* pRoot;

    void splitSamplesVec(int &featureIndex, 
                         float &threshold,
                         int* pSamples,
                         int  samplesSize,
                         int* pSamplesLeft,
                         int* pSampleLeftSize,
                         int* pSamplesRight,
                         int* pSampleRightSize,
                         Data &Data);

    void chooseBestSplitFeatures(Node* pNode,
                                 int* pSamples,
                                 int samplesSize,
                                 Data &Data);

    Node* constructNode(int* pSamples,
                        int samplesSize,
                        Data &trainData,
                        int depth);

    Node* constructNode(int* pSamples,
                        int sampleSize,
                        int* pSamplesRight,
                        int sampleRightSize,
                        int* pSamplesLeft,
                        int sampleLeftSize,
                        Data &trainData,
                        int depth);

public:
    /**
     *
     * @param criterion The function to measure the quality of a split.
     * Supported criteria are “gini” for the Gini impurity and “entropy"
     * for the information gain.
     * @param maxDepth The maximum depth of the tree. If 0, then nodes are
     * expanded until all leaves are pure or until
     * all leaves contain less than min_samples_split samples.
     * @param minSamplesSplit The minimum number of samples
     * required to split an internal node
     * @param minSamplesLeaf The minimum number of samples
     * required to be at a leaf node
     * @param sampleNum The number of samples to consider when constructing
     * tree.
     * @param maxFeatures The number of features to consider when looking for
     * the best split.
     */
    explicit DecisionTree(const string &criterion = "gini",
                          int maxDepth = -1,
                          int minSamplesSplit = 2,
                          int minSamplesLeaf = 1,
                          int sampleNum=-1,
                          const string &maxFeatures = "auto");

    void fit(Data &trainData);

    float computeProb(int sampleIndex, Data &Data);

    void predictProba(Data &Data, float* results);
};

#endif //RANDOMFOREST_DECISIONTREE_H
