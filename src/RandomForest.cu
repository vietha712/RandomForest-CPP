#include "RandomForest.h"


__device__ RandomForest::RandomForest(int nEstimators,
                           const char criterion[],
                           const char maxFeatures[],
                           int maxDepth,
                           int minSamplesSplit,
                           int minSamplesLeaf,
                           int eachTreeSamplesNum)
{
    this->nEstimators = nEstimators;

    this->criterion[0] = criterion[0];
    this->criterion[1] = criterion[1];
    this->criterion[2] = criterion[2];
    this->criterion[3] = criterion[3];

    this->maxFeatures[0] = maxFeatures[0];
    this->maxFeatures[1] = maxFeatures[1];
    this->maxFeatures[2] = maxFeatures[2];
    this->maxFeatures[3] = maxFeatures[3];

    this->maxDepth = maxDepth;
    this->minSamplesSplit = minSamplesSplit;
    this->minSamplesLeaf = minSamplesLeaf;
    this->eachTreeSamplesNum = eachTreeSamplesNum;
}

__device__ void RandomForest::fit(Data &trainData)
{
    DecisionTree results[MAX_TREE];
    for (int i = 0; i < nEstimators; i++)
    {
        DecisionTree tree(criterion, maxDepth, minSamplesSplit, minSamplesLeaf,
                          eachTreeSamplesNum, maxFeatures);
        tree.fit(trainData);
        //cout << "Fitted Tree: " << i << endl;
        results[i] = tree;
    }

    for (int i = 0; i < nEstimators; i++)
        decisionTrees[i] = results[i];
}

__device__ void RandomForest::norm(float* pTotal, int size)
{
    for (int i = 0; i < size; i++)
    {
        pTotal[i] /= nEstimators; 
    }
}

__device__ void RandomForest::predictProba(Data &Data, float* pResult)
{
    for (int i = 0; i < Data.getSampleSize(); i++)
        pResult[i] = 0.0;

    for (int i = 0; i < nEstimators; i++) {
        decisionTrees[i].predictProba(Data, pResult, Data.getSampleSize());
        //cout << "Predict Tree: " << i << endl;
    }

    norm(pResult, Data.getSampleSize());
}

__device__ void RandomForest::predict(Data &Data, int* pPredictedValue) {
    float results[TEST_SAMPLE_SIZE];

    predictProba(Data, results);

    //cout << "Sample size: " << Data.getSampleSize() << endl;

    for (int i = 0; i < Data.getSampleSize(); i++)
    {
        pPredictedValue[i] = (results[i] > 0.6) ? 1 : 0;
    }
}

__device__ void RandomForest::calculateMetrics(Data &testData, int* pPredictedValue)
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

__device__ float RandomForest::getAccuracy(void)
{
    return this->measurementRecords.accuracy;
}

__device__ float RandomForest::getRecall(void)
{
    return this->measurementRecords.recall;
}

__device__ float RandomForest::getPrecision(void)
{
    return this->measurementRecords.precision;
}

__device__ float RandomForest::getF1Score(void)
{
    return this->measurementRecords.f1Score;
}

__device__ float RandomForest::getFPR(void)
{
    return this->measurementRecords.falsePositiveRate;
}

__device__ float RandomForest::getFNR(void)
{
    return this->measurementRecords.falseNegativeRate;
}

__device__ float RandomForest::getTPR(void)
{
    return this->measurementRecords.truePositiveRate;
}

__device__ float RandomForest::getTNR(void)
{
    return this->measurementRecords.trueNegativeRate;
}
