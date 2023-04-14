//
// Created by Aster on 04/07/2018.
//

#include "../include/RandomForest.h"
#define MAX_TREE 250
void RandomForest::fit(Data &trainData)
{
    DecisionTree results[MAX_TREE];
    for (int i = 0; i < nEstimators; i++)
    {
        DecisionTree tree(criterion, maxDepth, minSamplesSplit, minSamplesLeaf,
                          eachTreeSamplesNum, maxFeatures);
        tree.fit(trainData);
        cout << "Fitted Tree: " << i << endl;
        results[i] = tree;
    }

    for (int i = 0; i < nEstimators; i++)
        decisionTrees[i] = results[i];
}

void RandomForest::norm(float* pTotal, int size) {
    for (int i = 0; i < size; i++) 
    {
        pTotal[i] /= nEstimators; 
    }
}

void vecAdd(vector<float> &total, vector<float> &part) {
    for (int i = 0; i < total.size(); i++) {
        total[i] += part[i];
    }
}

void RandomForest::predictProba(Data &Data, float* pResult)
{
    for (int i = 0; i < Data.getSampleSize(); i++)
        pResult[i] = 0.0;

    for (int i = 0; i < nEstimators; i++) {
        decisionTrees[i].predictProba(Data, pResult, Data.getSampleSize());
        cout << "Predict Tree: " << i << endl;
    }

    norm(pResult, Data.getSampleSize());
}

void RandomForest::predict(Data &Data, int* pPredictedValue) {
    float results[TEST_SAMPLE_SIZE];

    predictProba(Data, results);

    cout << "Sample size: " << Data.getSampleSize() << endl;

    for (int i = 0; i < Data.getSampleSize(); i++)
    {
        pPredictedValue[i] = (results[i] > 0.6) ? 1 : 0;
    }
}

void RandomForest::calculateMetrics(Data &testData, int* pPredictedValue)
{
    /* The label was placed in the last column of data set */
    this->measurementRecords.truePositive = 0;
    this->measurementRecords.falsePositive = 0;
    this->measurementRecords.trueNegative = 0;
    this->measurementRecords.falseNegative = 0;

    for (int i = 0; i < testData.getSampleSize(); i++)
    {
        unsigned int label = testData.readLabel(i);
        if (1 == label)
        {
            if (pPredictedValue[i] == label)
                this->measurementRecords.truePositive++;
            else
                this->measurementRecords.falseNegative++;
        }
        else
        {
            if (pPredictedValue[i] == label)
                this->measurementRecords.trueNegative++;
            else
                this->measurementRecords.falsePositive++;
        }
    }

    this->measurementRecords.accuracy = ((float)this->measurementRecords.truePositive + (float)this->measurementRecords.trueNegative) / (float)testData.getSampleSize();
    this->measurementRecords.recall   = (float)this->measurementRecords.truePositive / ((float)this->measurementRecords.truePositive + (float)this->measurementRecords.falseNegative);
    this->measurementRecords.precision = (float)this->measurementRecords.truePositive / ((float)this->measurementRecords.truePositive + (float)this->measurementRecords.falsePositive);
    this->measurementRecords.f1Score = 2 * ((this->measurementRecords.precision * this->measurementRecords.recall) / (this->measurementRecords.precision + this->measurementRecords.recall));
    this->measurementRecords.falsePositiveRate = (float)this->measurementRecords.falsePositive / ((float)this->measurementRecords.falsePositive + (float)this->measurementRecords.trueNegative);
    this->measurementRecords.truePositiveRate = (float)this->measurementRecords.truePositive / ((float)this->measurementRecords.truePositive + (float)this->measurementRecords.falseNegative);
    this->measurementRecords.falseNegativeRate = (float)this->measurementRecords.falseNegative / ((float)this->measurementRecords.truePositive + (float)this->measurementRecords.falseNegative);
    this->measurementRecords.trueNegativeRate = (float)this->measurementRecords.trueNegative / ((float)this->measurementRecords.trueNegative + (float)this->measurementRecords.falsePositive);
}

float RandomForest::getAccuracy(void)
{
    return this->measurementRecords.accuracy;
}

float RandomForest::getRecall(void)
{
    return this->measurementRecords.recall;
}

float RandomForest::getPrecision(void)
{
    return this->measurementRecords.precision;
}

float RandomForest::getF1Score(void)
{
    return this->measurementRecords.f1Score;
}

float RandomForest::getFPR(void)
{
    return this->measurementRecords.falsePositiveRate;
}

float RandomForest::getFNR(void)
{
    return this->measurementRecords.falseNegativeRate;
}

float RandomForest::getTPR(void)
{
    return this->measurementRecords.truePositiveRate;
}

float RandomForest::getTNR(void)
{
    return this->measurementRecords.trueNegativeRate;
}
