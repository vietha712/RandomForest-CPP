//
// Created by Aster on 02/07/2018.
//

#include "../include/DecisionTree.h"

int computeTrue(int* pSamples, int samplesSize, Data &Data) {
    int total = 0;
    for (int i = 0; i < samplesSize; i++)
    {
        total += Data.readLabel(pSamples[i]);
    }

    return total;
}

float computeTargetProb(int* pSamples, int samplesSize, Data &Data) {
    float num = 0;
    int total = 0;
    for (int i = 0; i < samplesSize; i++)
    {
        if (pSamples[i] != -1)
        {
            num += Data.readLabel(pSamples[i]);
            total++;
        }
    }
    return num / (total + 0.000000001);
}

//float getSize(vector<int> &samples) {
//    float num = 0;
//    for (auto i : samples) {
//        if (i != -1) { num++; }
//    }
//    return num;
//}

//float computeEntropy(vector<int> &samples, Data &Data) {
//    float trueProb = computeTargetProb(samples, Data);
//    return -1 * (trueProb * log2(trueProb)
//                 + (1 - trueProb) * log2((1 - trueProb)));
//}

float computeGini(int& sideTrue, int& sideSize) {
    float trueProb = (sideTrue * 1.0) / (sideSize + 0.00000001);
    return (1 - trueProb * trueProb - (1 - trueProb) * (1 - trueProb));
}

//float computeInformationGain(vector<int> &samples,
//                              vector<int> &pSamplesLeft,
//                              vector<int> &pSamplesRight,
//                              Data &Data) {
//    return -1 * computeEntropy(samples, Data)
//           + ( getSize(pSamplesLeft) / getSize(samples))
//             * computeEntropy(pSamplesLeft, Data)
//           + (getSize(pSamplesRight) / getSize(samples))
//             * computeEntropy(pSamplesRight, Data);
//}

float computeGiniIndex(int& leftTrue, int& leftSize,
                        int& rightTrue, int& rightSize) {
    float leftProb = (leftSize * 1.0) / (leftSize + rightSize);
    float rightprob = (rightSize * 1.0) / (leftSize + rightSize);
    return leftProb * computeGini(leftTrue, leftSize)
           + rightprob * computeGini(rightTrue, rightSize);
}

int _sqrt(int num) {
    return int(sqrt(num));
}

int _log2(int num) {
    return int(log2(num));
}

int _none(int num) {
    return num;
}

void DecisionTree::splitSamplesVec(int &featureIndex, float &threshold,
                                   int* pSamples,
                                   int  samplesSize,
                                   int* pSamplesLeft,
                                   int* pSampleLeftSize,
                                   int* pSamplesRight,
                                   int* pSampleRightSize,
                                   Data &Data)
{
    *pSampleLeftSize = 0;
    *pSampleRightSize = 0;

    for (int i = 0; i < samplesSize; i++)
    {
        int idxSampleRight = 0;
        int idxSampleLeft = 0;
        if (Data.readFeature(pSamples[i], featureIndex) > threshold) 
        {
            pSamplesRight[idxSampleRight] = pSamples[i];
            idxSampleRight++;
            *pSampleRightSize += 1;
        } 
        else 
        {
            pSamplesLeft[idxSampleLeft] = pSamples[i];
            idxSampleLeft++;
            *pSampleLeftSize += 1;
        }
    }
}

void sortByFeatures(vector<pair<int, float>>& samplesFeaturesVec,
                    int featureIndex, Data& data) {
    for (int i = 0; i < samplesFeaturesVec.size(); i++) {
        samplesFeaturesVec[i].second
                = data.readFeature(samplesFeaturesVec[i].first, featureIndex);
    }
    sort(samplesFeaturesVec.begin(), samplesFeaturesVec.end(), [](pair<int,
    float>& a, pair<int, float>& b) {
        return a.second < b.second;
    });
}

void DecisionTree::chooseBestSplitFeatures(Node* pNode,
                                           int* pSamples,
                                           int samplesSize,
                                           Data &Data)
{
    vector<int> featuresVec = Data.generateFeatures(this->maxFeatureFunc);
    int bestFeatureIndex = featuresVec[0];
    int samplesTrueNum = computeTrue(pSamples, samplesSize, Data);
    float minValue = 1000000000, bestThreshold = 0;
    float threshold = 0;
    int sampleIndex;

    vector<pair<int, float>> samplesFeaturesVec;
    samplesFeaturesVec.reserve(samplesSize);

    for (int i = 0; i < samplesSize; i++)
    {
        samplesFeaturesVec.emplace_back(pSamples[i], 0);
    }

    for (auto featureIndex : featuresVec) {
        sortByFeatures(samplesFeaturesVec, featureIndex, Data);
        int leftSize = 0, rightSize = samplesSize;
        int leftTrue = 0, rightTrue = samplesTrueNum;

        for (int index = 0; index < samplesFeaturesVec.size();)
        {
            sampleIndex = samplesFeaturesVec[index].first;
            threshold = samplesFeaturesVec[index].second;

            while ((index < samplesSize) && (samplesFeaturesVec[index].second <= threshold)) 
            {
                leftSize++;
                rightSize--;
                if (Data.readLabel(sampleIndex) == 1)
                {
                    leftTrue++;
                    rightTrue--;
                }
                index++;
                sampleIndex = samplesFeaturesVec[index].first;
            }
            if (index == samplesSize) { continue; }
            float value = criterionFunc(leftTrue, leftSize, rightTrue, rightSize);
            if (value <= minValue) {
                minValue = value;
                bestThreshold = threshold;
                bestFeatureIndex = featureIndex;
            }
        }
    }
    pNode->featureIndex = bestFeatureIndex;
    pNode->threshold = bestThreshold;
}

Node* DecisionTree::constructNode(int* pSamples,
                                  int sampleSize,
                                  Data &trainData,
                                  int depth)
{
    float targetProb = computeTargetProb(pSamples, sampleSize, trainData);
    int sampleLeft[50000];
    int sampleRight[50000];
    int* pSamplesLeft = sampleLeft;
    int* pSamplesRight = sampleRight;
    Node* pNode = new Node();

    pNode->depth = depth;
    pNode->prob = 0;
    int sampleLeftSize = 0;
    int sampleRightSize = 0;
    if ((targetProb == 0 )|| (targetProb == 1) ||
        (sampleSize <= minSamplesSplit) || (depth == maxDepth)) 
    {
        pNode->isLeaf = true;
        pNode->prob = targetProb;
    } 
    else 
    {
        chooseBestSplitFeatures(pNode, pSamples, sampleSize, trainData);

        splitSamplesVec(pNode->featureIndex,
                        pNode->threshold,
                        pSamples,
                        sampleSize,
                        pSamplesLeft,
                        &sampleLeftSize,
                        pSamplesRight,
                        &sampleRightSize,
                        trainData);

        if ((sampleLeftSize < minSamplesLeaf) || (sampleRightSize < minSamplesLeaf)) 
        {
            pNode->isLeaf = true;
            pNode->prob = targetProb;
        } 
        else 
        {
            pNode->pLeft = constructNode(pSamplesLeft, sampleLeftSize, trainData, depth + 1);
            pNode->pRight = constructNode(pSamplesRight, sampleRightSize, trainData, depth + 1);
        }
    }
    return pNode;

}

DecisionTree::DecisionTree(const string &criterion,
                           int maxDepth,
                           int minSamplesSplit,
                           int minSamplesLeaf,
                           int sampleNum,
                           const string &maxFeatures) {
    if (criterion == "gini") {
        this->criterionFunc = computeGiniIndex;
    } else if (criterion == "entropy") {
//        this->criterionFunc = computeInformationGain;
    } else {
        this->criterionFunc = computeGiniIndex;
    }

    if (maxFeatures == "auto" || maxFeatures == "sqrt") {
        this->maxFeatureFunc = _sqrt;
    } else if (maxFeatures == "log2") {
        this->maxFeatureFunc = _log2;
    } else {
        this->maxFeatureFunc = _none;
    }
    this->sampleNum = sampleNum;
    this->maxDepth = maxDepth;
    this->minSamplesSplit = minSamplesSplit;
    this->minSamplesLeaf = minSamplesLeaf;
}

void DecisionTree::fit(Data &trainData) {
    int pSamples[50000];
    pSamples = trainData.generateSample(this->sampleNum);
    pRoot = constructNode(pSamples, this->sampleNum, trainData, 0);
}

float DecisionTree::computeProb(int sampleIndex, Data &Data) {
    auto node = pRoot;
    while (!node->isLeaf) {
        if (Data.readFeature(sampleIndex, node->featureIndex) > node->threshold) 
        {
            node = node->pRight;
        } else {
            node = node->pLeft;
        }
    }
    return node->prob;
}

void DecisionTree::predictProba(Data &Data, vector<float> &results) {
    for (int i = 0; i < results.size(); i++) {
        results[i] += computeProb(i, Data);
    }
}
