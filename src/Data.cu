#include "Data.h"
#include "Utilities.cuh"
#include "cuda_runtime.h"
#include "Utilities.cuh"

__host__ vector<string> splitBySpace(string &sentence)
{
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

__host__ __device__ Data::Data(bool isTrain, int size, int featuresSize)
{
    this->featureSize = featuresSize;
    this->samplesSize = size;

    this->isTrain = isTrain;
}

__host__ __device__ Data::Data(){}

__host__ void Data::read(const string &filename) {
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

        for (int j = 0; j < this->featureSize; j++)
        {
            sample[j] = stod(results[j]);
            d_pFeatures[featureSize * i + j] = sample[j]; /* Copy value from host to device */
        }

        if (this->isTrain)
        {
            /* The last member store the label [0, 1] */
            //target.push_back(atoi(results[this->featureSize].c_str()));
            d_pTarget[i] = atoi(results[this->featureSize].c_str()); /* Value to device */
        }
        d_pSampleVector[i] = i;
    }
    inputFile.close();

    for (int i = 0; i < featureSize; i++)
    {
        d_pfeaturesVec[i] = i; /* Value to device */
    }
}

__host__ void Data::read(const string &filename, int idx[]) {
    ifstream inputFile;
    inputFile.open(filename.c_str());

    if (!inputFile.is_open()) { cout << "Failed Open" << endl; }
    string str;
    vector<float> sample(this->featureSize, 0);
    for (int i = 0; i < this->samplesSize; i++)
    {
        (void)getline(inputFile, str);
        auto results = splitBySpace(str);
        
        for (int j = 0; j < this->featureSize; j++)
        {
            sample[j] = stod(results[idx[j]]);
        }
        gpuErrchk(cudaMemcpy(&d_pFeatures[featureSize * i], &sample[0], featureSize * sizeof(float), cudaMemcpyHostToDevice));
        if (this->isTrain)
        {
            /* The last member store the label [0, 1] */
            int tmpBuffer = atoi(results[this->featureSize].c_str()); /* Value to device */
            //d_pTarget[i] = atoi(results[this->featureSize].c_str()); /* Value to device */
            gpuErrchk(cudaMemcpy(&d_pTarget[i], &tmpBuffer, sizeof(int), cudaMemcpyHostToDevice));
        }
        //d_pSampleVector[i] = i;
        gpuErrchk(cudaMemcpy(&d_pSampleVector[i], &i, sizeof(int), cudaMemcpyHostToDevice));
        
    }
    inputFile.close();

    for (int i = 0; i < featureSize; i++)
    {
        //d_pfeaturesVec[i] = i; /* Value to device */
        gpuErrchk(cudaMemcpy(&d_pfeaturesVec[i], &i, sizeof(int), cudaMemcpyHostToDevice));
    }
}

__device__ float Data::readFeature(int sampleIndex, int featureIndex) {
    return d_pFeatures[this->featureSize * sampleIndex + featureIndex];
}

__device__ int Data::readLabel(int sampleIndex) {
    return d_pTarget[sampleIndex];
}

__host__ __device__ int Data::getSampleSize() {
    return samplesSize;
}

__host__ __device__ int Data::getFeatureSize() {
    return featureSize;
}

//vector<int> Data::generateSample(int &num) {
//    if (num == -1) {
//        return pSamples;
//    } else {
//        random_shuffle(pSamples.begin(), pSamples.end());
//        return vector<int>(pSamples.begin(), pSamples.begin() + num);
//    }
//}

__device__ void Data::generateSample(int* pSamples, int &num) {
    if (num == -1)
    {
        return;
    } 
    else 
    {
        for (int i = 0; i < samplesSize; i++)
            pSamples[i] = d_pSampleVector[i];
        randomize(pSamples, num);
    }
}

//vector<int> Data::generateFeatures(function<int(int)> &func) {
//    int m = func(this->getFeatureSize());
    //random_shuffle(featuresVec.begin(), featuresVec.end());
    //return vector<int>(featuresVec.begin(), featuresVec.begin() + m);
//}

__device__ void Data::generateFeatures(maxFeatureFunc fPtr, int* pFeatVector)
{
    int tempVec[FEATURE_SIZE];
    int m = fPtr(this->getFeatureSize());

    for (int i = 0; i < featureSize; i++)
        tempVec[i] = d_pFeatures[i];

    randomize(tempVec, this->getFeatureSize());
    for (int i = 0; i < featureSize; i++)
        d_pFeatures[i] = tempVec[i];

    for (int i = 0; i < m; i++)
        pFeatVector[i] = tempVec[i];
}

//__device__ void Data::sortByFeature(vector<int> &pSamples, int featureIndex)
//{
//    sort(pSamples.begin(), pSamples.end(), [&](int a, int b) {
//        return this->readFeature(a, featureIndex) <
//               this->readFeature(b, featureIndex);
//    });
//}

__host__ void Data::allocateMemForData(void)
{
    gpuErrchk(cudaMalloc((void **)&d_pFeatures, this->featureSize * this->samplesSize * sizeof(float)));
    gpuErrchk(cudaMalloc((void **)&d_pTarget, this->samplesSize * sizeof(int)));
    gpuErrchk(cudaMalloc((void **)&d_pSampleVector, this->samplesSize * sizeof(int)));
    gpuErrchk(cudaMalloc((void **)&d_pfeaturesVec, this->featureSize * sizeof(int)));
    printf("allocateMemForData - ok\n");
}
