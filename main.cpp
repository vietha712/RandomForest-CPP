#include "include/DecisionTree.h"
#include "include/Data.h"
#include "include/RandomForest.h"

int main() {
    /* This is for testing function read by index. Using all the index to simulate normal read data */
    vector<int> vect1;
    for (int i = 0; i < FEATURE_SIZE; i++)
    {
        vect1.push_back(i);
    }
    string a = "../data/std_scale_dos.txt";

    Data trainData(true, TRAIN_SAMPLE_SIZE, vect1.size());
    trainData.read(a, vect1);
    cout << "Read data success" << endl;

    RandomForest randomForest(200, "gini", "auto ", -1, 2, 1, 10, 1);
    cout << "RandomForest success" << endl;
    randomForest.fit(trainData);
    cout << "Training data success" << endl;

    Data testData(true, TEST_SAMPLE_SIZE, vect1.size());
    testData.read("../data/std_scale_dos_test.txt", vect1); // Failed here
    cout << "Read test data success: " << endl;

    float proba[TEST_SAMPLE_SIZE];
    randomForest.predictProba(testData, proba);
    //writeDataToCSV(proba, testData, "../results/trainResults.csv", false);
    cout << "Predict probabilities success" << endl;
    
    int predictedLabel[TEST_SAMPLE_SIZE];
    randomForest.predict(testData, predictedLabel);
    //writeDataToCSV(predictedLabel, testData, "../results/testResults.csv", false);
    cout << "Predict success" << endl;

    randomForest.calculateMetrics(testData, predictedLabel);
    printf("accuracy: %f\n", randomForest.getAccuracy());
    printf("recall: %f\n", randomForest.getRecall());
    printf("precision: %f\n", randomForest.getPrecision());
    printf("f1 score: %f\n", randomForest.getF1Score());
    printf("False Positive Rate: %f\n", randomForest.getFPR());
    printf("False Negative Rate: %f\n", randomForest.getFNR());
    printf("True Positive Rate: %f\n", randomForest.getTPR());
    printf("True Negative Rate: %f\n", randomForest.getTNR());


    return 0;
}
