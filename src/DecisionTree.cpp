//
// Created by Aster on 02/07/2018.
//

#include "../include/DecisionTree.h"
#include "../include/utils.h"

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
        int samplesIndex = pSamples[i];
        if (Data.readFeature(samplesIndex, featureIndex) > threshold) 
        {
            pSamplesRight[*pSampleRightSize] = samplesIndex;
            *pSampleRightSize += 1;
        } 
        else 
        {
            pSamplesLeft[*pSampleLeftSize] = samplesIndex;
            *pSampleLeftSize += 1;
        }
    }
}

void sortByFeatures(int* pSampleFeatVecIdx, float* pSamplesFeaturesVec, int sampleVectorSize,
                    int featureIndex, Data& data) 
{
    for (int i = 0; i < sampleVectorSize; i++) 
    {
        pSamplesFeaturesVec[i] = data.readFeature(pSampleFeatVecIdx[i], featureIndex);
    }
    quickSortWithIdx(pSamplesFeaturesVec, pSampleFeatVecIdx, 0, sampleVectorSize - 1);

}

void DecisionTree::chooseBestSplitFeatures(Node* pNode,
                                           int* pSamples,
                                           int samplesSize,
                                           Data &Data)
{
    int   featuresVec[FEATURE_SIZE];
    int   bestFeatureIndex = featuresVec[0];
    int   samplesTrueNum;;
    float minValue = 1000000000, bestThreshold = 0;
    float threshold = 0;
    int   sampleIndex;
    int   sampleVectorIdx[TRAIN_SAMPLE_SIZE];
    float sampleVector[TRAIN_SAMPLE_SIZE];

    Data.generateFeatures(this->maxFeatureFunc, featuresVec);

    samplesTrueNum = computeTrue(pSamples, samplesSize, Data);

    for (int i = 0; i < samplesSize; i++)
    {
        sampleVectorIdx[i] = i;
        sampleVector[i] = 0.0;
    }

    for (auto featureIndex : featuresVec) 
    {
        sortByFeatures(sampleVectorIdx, sampleVector, samplesTrueNum, featureIndex, Data);

        int leftSize = 0, rightSize = samplesSize;
        int leftTrue = 0, rightTrue = samplesTrueNum;

        for (int index = 0; index < samplesTrueNum;)
        {
            sampleIndex = sampleVectorIdx[index];
            threshold = sampleVector[index];

            while ((index < samplesSize) && (sampleVector[index] <= threshold)) 
            {
                leftSize++;
                rightSize--;
                if (Data.readLabel(sampleIndex) == 1)
                {
                    leftTrue++;
                    rightTrue--;
                }
                index++;
                sampleIndex = sampleVectorIdx[index];
            }

            if (index == samplesSize) { continue; }

            float value = criterionFunc(leftTrue, leftSize, rightTrue, rightSize);
            if (value <= minValue) 
            {
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
    int sampleLeft[TRAIN_SAMPLE_SIZE];
    int sampleRight[TRAIN_SAMPLE_SIZE];
    Node* pNode = new Node();

    pNode->depth = depth;
    pNode->prob = 0;
    int sampleLeftSize = 0;
    int sampleRightSize = 0;
    if ((targetProb == 0 )|| (targetProb == 1) || (sampleSize <= minSamplesSplit) || (depth == maxDepth)) 
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
                        &sampleLeft[0],
                        &sampleLeftSize,
                        &sampleRight[0],
                        &sampleRightSize,
                        trainData);

        if ((sampleLeftSize < minSamplesLeaf) || (sampleRightSize < minSamplesLeaf)) 
        {
            pNode->isLeaf = true;
            pNode->prob = targetProb;
        } 
        else 
        {
            pNode->pLeft = constructNode(&sampleLeft[0], sampleLeftSize, trainData, depth + 1);
            pNode->pRight = constructNode(&sampleRight[0], sampleRightSize, trainData, depth + 1);
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
    int pSamples[TRAIN_SAMPLE_SIZE];
    trainData.generateSample(pSamples, this->sampleNum);
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

void DecisionTree::predictProba(Data &Data, float* pResults, int resultSize) {
    for (int i = 0; i < resultSize; i++) {
        pResults[i] += computeProb(i, Data);
    }
}
