//
// Created by Aster on 04/07/2018.
//

#include "../include/RandomForest.h"

void RandomForest::fit(Data &trainData) {
    ThreadPool pool(this->nJobs);
    std::vector<std::future<DecisionTree>> results;
    for (int i = 0; i < nEstimators; i++) {
        results.emplace_back(pool.enqueue([&, i] {
            DecisionTree tree(criterion, maxDepth, minSamplesSplit, minSamplesLeaf,
                              eachTreeSamplesNum, maxFeatures);
            tree.fit(trainData);
            cout << "Fitted Tree: " << i << endl;
            return tree;
        }));
    }

    for (auto &&each : results) {
        decisionTrees.push_back(each.get());
    }
}

void RandomForest::norm(vector<float> &total) {
    for (float &i : total) { i /= nEstimators; }
}

void vecAdd(vector<float> &total, vector<float> &part) {
    for (int i = 0; i < total.size(); i++) {
        total[i] += part[i];
    }
}

vector<float> RandomForest::predictProba(Data &Data) {
    ThreadPool pool(this->nJobs);
    vector<future<vector<float>>> poolResults;
    for (int i = 0; i < nEstimators; i++) {
        poolResults.emplace_back(pool.enqueue([&, i] {
            vector<float> results(Data.getSampleSize(), 0);
            decisionTrees[i].predictProba(Data, results);
            cout << "Predict Tree: " << i << endl;
            return results;
        }));
    }
    vector<float> results(Data.getSampleSize(), 0);
    for (auto &&each : poolResults) {
        auto tmpResults = each.get();
        vecAdd(results, tmpResults);
    }
    norm(results);
    return results;
}

void RandomForest::predict(Data &Data, vector<int> &predictedValue) {
    vector<float> results;

    results = predictProba(Data);

    cout << "Sample size: " << Data.getSampleSize() << endl;

    for (int i = 0; i < Data.getSampleSize(); i++)
    {
        predictedValue[i] = (results[i] > 0.6) ? 1 : 0;
    }
}

void RandomForest::calculateMetrics(Data &testData, vector<int> &predictedValue)
{
    /* The label was placed in the last column of data set */
    this->measurementRecords.truePositive = 0;
    this->measurementRecords.falsePositive = 0;
    this->measurementRecords.trueNegative = 0;
    this->measurementRecords.falseNegative = 0;

    for (int i = 0; i < testData.getSampleSize(); i++)
    {
        unsigned int label = testData.readTarget(i);
        if (1 == label)
        {
            if (predictedValue[i] == label)
                this->measurementRecords.truePositive++;
            else
                this->measurementRecords.falseNegative++;
        }
        else
        {
            if (predictedValue[i] == label)
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
