
//STD Dependencies
#include <map>
#include <vector>
#include <cmath>
//#include "ue3.h"

//STD Dependencies
#include <iostream>
#include <fstream>
#include <chrono>
#include <sys/stat.h>
#include <map>
#include <vector>
#include <cmath>

//Plot Dependencies
#include "gnuplot-iostream.h"

//OpenCV Dependencies
#include<opencv2/opencv.hpp> 
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include<opencv2/ml.hpp>
#include<opencv2/dpm.hpp>
#include <opencv2/highgui/highgui.hpp>
#include<opencv/highgui.h>


/****************************************************************************************\
*                                 Global Variables		                                 *
\****************************************************************************************/

#define RandomForest cv::ml::RTrees
#define PtrRandomForest cv::Ptr<RandomForest>

#define SVM cv::ml::SVM
#define PtrSVM cv::Ptr<SVM>

#define ANN cv::ml::ANN_MLP
#define PtrANN cv::Ptr<ANN>

// define paths
const std::string IMAGE_DIR_GTSRB = "TrainIJCNN2013/";
const std::string HOG2_DIR = "HOG_02/";
const std::string PCA_FILENAME = "pca_transformation.json";
const std::string RF_FILENAME = "rf_classif.json";
const std::string SVM_FILENAME = "svm_classif.json";
const std::string ANN_FILENAME = "ann_classif.json";
const std::string PROJ_RAWFEATURES_FILENAME = "proj_rawfeatures.json";
const std::string FEAT_LABELS_FILENAME = "labels_features.json";

const std::string TEST_RAWFEATURES_FILENAME = "test_rawfeatures.json";

const std::string IMAGE_DIR_TEST = "Test/";
const std::string HOG2_DIR_TEST = "HOG_02_Test/";
const std::string GT_TEST = "GT_testSet.csv";


// create categories of traffic signs
const std::vector<unsigned int> PROHIBITORY = { 0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 16 };
const std::vector<unsigned int> MANDATORY = { 33, 34, 35, 36, 37, 38, 39, 40 };
const std::vector<unsigned int> DANGER = { 11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31 };
const std::vector<unsigned int> ALL;


// define the relevant classes
// we just consider a binary classification problem, so only two classes are relevant!
// example hard: 1 - 30er, 2 - 50er
// example easy: 1 - 30er, 38 - keep right

const std::vector<unsigned int> CONSIDERED_CLASS_IDs = { 0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 16, 11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 33, 34, 35, 36, 37, 38, 39, 40};//{11, 14, 33, 41};

cv::PCA pca;

cv::HOGDescriptor hog(
	cv::Size(64, 64),
	cv::Size(16, 16),
	cv::Size(8, 8),
	cv::Size(8, 8),
	9, 1, -1.0, 0, 0.2, false, 64);

cv::HOGDescriptor MNIST_HOG_1(
	cv::Size(32, 32), //WinSize
	cv::Size(16, 16), //BlockSize
	cv::Size(16, 16), //BlockStride
	cv::Size(16, 16), //CellSize
	12, //No Bins
	1,
	-2.0, //winSigma
	0, 0.2, false, 64, 
	true //signedGradient
);

cv::HOGDescriptor MNIST_HOG_2(
	cv::Size(32, 32), //WinSize
	cv::Size(16, 16), //BlockSize
	cv::Size(8, 8), //BlockStride
	cv::Size(8, 8), //CellSize
	12, //No Bins
	1,
	-2.0, //winSigma
	0, 0.2, false, 64,
	true //signedGradient
);

cv::HOGDescriptor MNIST_HOG_3(
	cv::Size(32, 32), //WinSize
	cv::Size(8, 8), //BlockSize
	cv::Size(4, 4), //BlockStride
	cv::Size(4, 4), //CellSize
	12, //No Bins
	1,
	-2.0, //winSigma
	0, 0.2, false, 64,
	true //signedGradient
);

int PCA_AxesNumber = 10;
const double trainPercentage = 0.8;
const double activeVarPercentage = 0.06377;

/****************************************************************************************\
*                    Required Function Declarations                                      *
\****************************************************************************************/
void visualizePCA(const cv::Mat &features, const cv::Mat &labels);

/****************************************************************************************\
*                   Construct Data Structures - Functions                                *
\****************************************************************************************/

// define data structures
struct imageLabelGTSRB {
	std::string filename;
	unsigned int classID;
	imageLabelGTSRB(std::string file, unsigned int id) : filename(file), classID(id) {}
};

struct evalStruct {
	double avgTime;
	double evalPerformance;

	evalStruct(double _avgTime, double _evalPerformance) : avgTime(_avgTime), evalPerformance(_evalPerformance) {}
	evalStruct() : avgTime(0.0), evalPerformance(0.0) {}
};

// check if file exists
bool fileExists(const std::string fileName) {
	struct stat stFileInfo;
	const int intStat = stat(fileName.c_str(), &stFileInfo);
	return (intStat == 0);
}

// check if classID is relevant
// i.e. return true if vector is empty or classID is contained in the vector
bool isClassRelevant(const std::vector<unsigned int> relevantClasses, const unsigned int classID) {
	return (relevantClasses.empty() || (std::find(relevantClasses.begin(), relevantClasses.end(), classID) != relevantClasses.end()));
}

// read the GTSRB dataset
void readDataSet(std::vector<imageLabelGTSRB> &records, std::vector<unsigned int> &nSamplesPerClass,
	std::vector<unsigned int> relevantClasses = ALL) {
	unsigned int nSamples = 0;
	for (unsigned int c = 0;; ++c) {

		if (!isClassRelevant(relevantClasses, c)) {

			// check whether there ought to be more relevant classes
			if (c > *std::max_element(std::begin(relevantClasses), std::end(relevantClasses))) {
				// end completely
				break;
			}
			else {
				// if class is not relevant we can just skip it
				nSamplesPerClass.push_back(0);
				continue;
			}
			break;
		}
		bool foundFileForClass = false;

		for (int t = 0;; ++t) {

			bool foundFileForTrack = false;
			// +=4: we won't use every single frame of each track, just every 4th
			// we have plenty of data and want a subset of good diversity
			for (int e = 0;; e += 4) {

				char fileName[32];
				sprintf_s(fileName, "%05d/%05d_%05d.ppm", c, t, e);
				std::string filePath = IMAGE_DIR_GTSRB + fileName;

				if (fileExists(filePath)) {
					foundFileForClass = true;
					foundFileForTrack = true;
					nSamples++;
					records.push_back(imageLabelGTSRB(filePath, c));
				}
				else break;
			}
			if (false == foundFileForTrack) {
				std::cout << "[info]\tfound " << nSamples << " samples of class " << c << "." << std::endl;
				nSamplesPerClass.push_back(nSamples);
				nSamples = 0;
				break;
			}
		}
		if (false == foundFileForClass) break;
	}
}

// this methods splits the dataset given by records (and further specified by nSamplesPerClass)
// into training and validation records. trainRatio specifies the ratio of training data.
// e.g. trainRatio = 0.75 means that 75% of records will be training, 25% validation data.
// optionally, you can also retrieve the number of samples per class in the training and validation
// set by specifying nSamplesPerClassTrain and nSamplesPerClassVal respectively.
// > note that this method is deterministic and gives equal results for equal inputs!
void splitDataSet(std::vector<imageLabelGTSRB> records, std::vector<unsigned int> nSamplesPerClass,
	std::vector<imageLabelGTSRB> &trainRecs, std::vector<imageLabelGTSRB> &valRecs, double trainRatio,
	std::vector<unsigned int> &nSamplesPerClassTrain, std::vector<unsigned int> &nSamplesPerClassVal)
{
	unsigned int offset = 0;
	// for each class separately
	for (unsigned int c = 0; c < nSamplesPerClass.size(); ++c) {

		// compute critical indices
		unsigned int lastTrainIndex = offset + static_cast<unsigned int>(trainRatio * nSamplesPerClass[c]);
		unsigned int lastValIndex = offset + nSamplesPerClass[c];

		// get the number of samples in each set right
		unsigned int nSamplesTrain = static_cast<unsigned int>(trainRatio * nSamplesPerClass[c]);
		unsigned int nSamplesVal = nSamplesPerClass[c] - nSamplesTrain;
		nSamplesPerClassTrain.push_back(nSamplesTrain);
		nSamplesPerClassVal.push_back(nSamplesVal);

		// insert elements into other vectors accordingly
		trainRecs.insert(trainRecs.end(), records.begin() + offset, records.begin() + lastTrainIndex);
		valRecs.insert(valRecs.end(), records.begin() + lastTrainIndex, records.begin() + lastValIndex);

		// remember current position
		offset += nSamplesPerClass[c];
	}
	std::cout << "[debug]\t#samples train: " << trainRecs.size() << std::endl;
	std::cout << "[debug]\t#samples val:   " << valRecs.size() << std::endl;
	std::cout << "[debug]\trealized ratio: " << 1.*trainRecs.size() / (trainRecs.size() + valRecs.size()) << std::endl;
}

void splitDataSet(std::vector<imageLabelGTSRB> records, std::vector<unsigned int> nSamplesPerClass,
	std::vector<imageLabelGTSRB> &trainRecs, std::vector<imageLabelGTSRB> &valRecs, double trainRatio = 0.75) {
	std::vector<unsigned int> nSamplesPerClassTrain;
	std::vector<unsigned int> nSamplesPerClassVal;

	splitDataSet(records, nSamplesPerClass, trainRecs, valRecs, trainRatio, nSamplesPerClassTrain, nSamplesPerClassVal);
}

/****************************************************************************************\
*                   Feature Extraction - Functions                                       *
\****************************************************************************************/

//- return the dimension of the HOG features you use
unsigned int getFeatureVectorDimension(){
	return static_cast<unsigned int>(hog.getDescriptorSize());
}

//- compute the hog features for a given roi and fill the values into the feature vector
void computeFeatures(cv::Mat roi, std::vector<float> &features)
{
	//resize ROI
	cv::Mat roi_modif;
	cv::resize(roi, roi_modif, hog.winSize);

	//compute HOG
	hog.compute(roi_modif, features);
}

//- compute the hog features for a given roi and fill the values into the feature vector
void computeFeatures(cv::Mat roi, std::vector<float> &features, const cv::HOGDescriptor& hog_d)
{
	//resize ROI
	cv::Mat roi_modif;
	cv::resize(roi, roi_modif, hog_d.winSize);

	//compute HOG
	hog_d.compute(roi_modif, features);
}


// this method computes the features for a given dataset
// it organizes the features and labels conveniently in cv::Mat format used for most OpenCV classifiers
void computeFeaturesToMat(std::vector<imageLabelGTSRB> records, cv::Mat &features, cv::Mat &labels) {

	// initialize data 
	// in the mat, each row is a sample vector
	// each column represents one feature dimension
	unsigned int rows = records.size();
	unsigned int cols = MNIST_HOG_1.getDescriptorSize() + MNIST_HOG_2.getDescriptorSize() +
		MNIST_HOG_3.getDescriptorSize() + 1568;
	features = cv::Mat(rows, cols, CV_32FC1);
	labels = cv::Mat(rows, 1, CV_32FC1);

	// probably we need to deal with mapping the labels to contiguous values
	// let's postpone this. idea is to deliver a map, which can than be used at classification time.
	std::cout << "[info]\tcomputing features " << std::flush;
	unsigned int step_size = rows / 10;
	unsigned int step = step_size;

	// for each sample in records
	for (unsigned int s = 0; s < rows; ++s) {

		if (s >= step) { // stupid visualization of progress
			step += step_size;
			std::cout << "." << std::flush;
		}

		// read image
		cv::Mat roi = cv::imread(records[s].filename);

		// resize image and compute features
		std::vector<float> descriptors;
		int weight = 1;
		for (auto& hog_d : { MNIST_HOG_1, MNIST_HOG_2, MNIST_HOG_3 }) {
			std::vector<float> descriptors_aux;
			computeFeatures(roi, descriptors_aux, hog_d);
			std::transform(descriptors_aux.begin(), descriptors_aux.end(), descriptors_aux.begin(),
				[weight](float d) -> float { return d * weight; });
			descriptors.insert(descriptors.end(), descriptors_aux.begin(), descriptors_aux.end());
			weight *= 2;
		}

		//Look for the HOG2 preprocesed features
		std::string HOG2_name = HOG2_DIR + records[s].filename.substr(IMAGE_DIR_GTSRB.size()).substr(0, 
			records[s].filename.substr(IMAGE_DIR_GTSRB.size()).size()-4) + ".txt";

		//Append the HOG_2 features to the vector
		std::ifstream hog2_file(HOG2_name);
		while (hog2_file.good()) {
			std::string tmp;
			std::getline(hog2_file, tmp);
			descriptors.push_back(std::stof(tmp));
		}
		hog2_file.close();

		// copy descriptors to features mat
		assert(cols == descriptors.size());
		for (unsigned int f = 0; f < cols; ++f)
			features.at<float>(s, f) = descriptors[f];

		// set labels
		labels.at<float>(s) = static_cast<float>(records[s].classID);
	}
	std::cout << " done!" << std::endl;
}

void prepareLabelsANN(cv::Mat& preparedLabels, std::map<int, int>& col_to_label, const cv::Mat& labels) {
	
	//preparedLabels needs to be a CV_32FC1
	preparedLabels.convertTo(preparedLabels, CV_32FC1);

	preparedLabels = 
		cv::Mat::ones(cv::Size(CONSIDERED_CLASS_IDs.size(),labels.rows), CV_32FC1); //set matrix to -1;
	preparedLabels = (-1)*preparedLabels;

	col_to_label.clear();
	std::map<int, int> label_to_col;
	for (auto &id : CONSIDERED_CLASS_IDs) { //Keep a key-value map of the columns and the corresponding classes
		int col = col_to_label.size();
		col_to_label[col] = id;
		label_to_col[id] = col;
	}

	for (int index = 0; index < labels.rows; index++) {
		int this_label = labels.at<int>(index, 0);
		preparedLabels.at<float>(index, label_to_col.at(this_label)) = 1;
	}
	//std::cout << preparedLabels << std::endl;
	//std::cout << labels << std::endl;
	//std::cout << "Here" << std::endl;
}

//Transforms the response of a ANN to its corresponding labels. The procedure is made by taking the highest response
//in the output vector of the ANN and assigning it to the corresponding label. (arg_max(Z))
void reconstructLabelsANN(cv::Mat& inferedLabels, const std::map<int, int>& col_to_label, const cv::Mat& resultANN) {
	
	inferedLabels.convertTo(inferedLabels, CV_32S); //The labels array must be an integer array

	for (int index = 0; index < resultANN.rows; index++) {

		cv::Mat row = resultANN.row(index);
		double min = 0, max = 0;
		cv::Point minLoc, maxLoc;
		minMaxLoc(row, &min, &max, &minLoc, &maxLoc);

		inferedLabels.push_back(col_to_label.at(maxLoc.x));
	}
}

// this method is suitable for splitting a string into a vector of substrings, divided by a delimiter character
// source: http://stackoverflow.com/questions/236129/split-a-string-in-c
// usage: create a vector of strings, then call split(string, delimiter, vector);
void split(const std::string &s, char delim, std::vector<std::string> &elems) {
	std::stringstream ss(s);
	std::string item;
	while (std::getline(ss, item, delim)) {
		elems.push_back(item);
	}
}

/****************************************************************************************\
*                           ML Functions					                             *
\****************************************************************************************/

// prints the obtained confusion matrix of the algorithm and the total precision
void evalClassif(const cv::Mat& testResponse, float& count, float& accuracy, const cv::Mat& testLabels) {

	std::map<int, int> confMatrixLegend;
	for (auto &id : CONSIDERED_CLASS_IDs) confMatrixLegend[id] = confMatrixLegend.size(); 

	cv::Mat confusionMatrix = cv::Mat::zeros(cv::Size(confMatrixLegend.size(), confMatrixLegend.size()), CV_32S);

	for (int i = 0; i < testResponse.rows; i++) {
		//std::cout << "Test Response , TestLabels " << testResponse.at<float>(i, 0) << " " << testLabels.at<int>(i, 0) << std::endl;
		if (testResponse.at<float>(i, 0) == testLabels.at<int>(i, 0)) count = count + 1;
		int x_coord = confMatrixLegend.at(testLabels.at<int>(i, 0));
		int y_coord = confMatrixLegend.at(testResponse.at<float>(i, 0));
		confusionMatrix.at<int>(x_coord, y_coord) += 1;
	}
	accuracy = (count / testResponse.rows) * 100;
	std::cout << "\n" << confusionMatrix << std::endl;
	std::cout << "\nLeyend : " << std::endl;

	for (int i = 0; i < confMatrixLegend.size(); i++) 
		std::cout << "Row " << i << " -> class " << CONSIDERED_CLASS_IDs.at(i) << std::endl;

	std::cout << std::endl;
	std::cout << "Classifier Accuracy : " << accuracy << "\n"  << std::endl;
}

//- adjust the model to your chosen classifier type
double evaluateClassifier(const PtrRandomForest &classifier, const cv::Mat &features, const cv::Mat &labels) {

	if (features.empty()) {
		std::cout << "[error]\tevaluateClassifier(): given feature vector is empty." << std::endl;
		return 0.0;
	}

	cv::Mat predictions;
	classifier->predict(features, predictions);

	float count = 0.0;
	float accuracy = 0.0;

	evalClassif(predictions, count, accuracy, labels);
	return 0;
}

std::map<int, int> col_to_label;
template<typename T>
double evalClassifier(const T &classifier, const cv::Mat &features, const cv::Mat &labels) {

	if (features.empty()) {
		std::cout << "[error]\tevaluateClassifier(): given feature vector is empty." << std::endl;
		return 0.0;
	}

	cv::Mat predictions;
	classifier->predict(features, predictions);

	if (typeid(T) == typeid(PtrANN)) {
		cv::Mat predictionsReconstructed;
		reconstructLabelsANN(predictionsReconstructed, col_to_label, predictions);
		predictions = predictionsReconstructed.clone();
		predictions.convertTo(predictions, CV_32FC1);
	}

	float count = 0.0;
	float accuracy = 0.0;

	evalClassif(predictions, count, accuracy, labels);
	return accuracy;
}



/****************************************************************************************\
*                           Visualization Functions 		                             *
\****************************************************************************************/
// this function displays the first two principal components on a gnuplot
void visualizePCA(const cv::Mat &features, const cv::Mat &labels) {

	Gnuplot gp;
	// assert features.rows == labels.rows
	if (features.rows != labels.rows) {
		std::cout << "[error] cannot display pca. (dimension mismatch.)" << std::endl;
		return;
	}

	float current_label = -1.;
	std::vector<std::pair<double, double> > xy_pts;
	gp << "set xrange [-5:5]\nset yrange [-5:5]\n";
	gp << "plot";
	gp << std::fixed << std::setprecision(0); // for nice display of float labels

	for (int i = 0; i < features.rows; i++)
	{
		if (labels.at<float>(i, 0) != current_label) {

			// plot vector if not empty (will only be empty first time)
			if (!xy_pts.empty())
				gp << gp.file1d(xy_pts) << "with points title 'class ID: " << current_label << "',";

			// create shiny new vector
			xy_pts = std::vector<std::pair<double, double> >();

			// remember the new label
			current_label = labels.at<float>(i, 0);
		}

		// push the data point to the vector
		xy_pts.push_back(std::make_pair(
			features.at<float>(i, 0),
			features.at<float>(i, 1)));
	}
	// and plot the last vector
	gp << gp.file1d(xy_pts) << "with points title 'class ID: " << current_label << "'" << std::endl;
	std::cout << "[info]\tclose the gnuplot window to continue." << std::endl;
}

// main method
// takes care of program flow
int main(int argc, char* argv[]){

	std::cout << MNIST_HOG_1.getDescriptorSize() << " " << MNIST_HOG_1.getDescriptorSize()/12 << std::endl;
	std::cout << MNIST_HOG_2.getDescriptorSize() << " " << MNIST_HOG_2.getDescriptorSize() / 12 << std::endl;
	std::cout << MNIST_HOG_3.getDescriptorSize() << " " << MNIST_HOG_3.getDescriptorSize() / 12 << std::endl;

	cv::Mat trainingFeatures;
	cv::Mat trainingLabels;

	cv::Mat valLabels;
	cv::Mat valFeatures;

	bool calculateStructure = false;
	if(calculateStructure){

		/*** READING INPUT DATA ***/
		std::cout << "[info]\treading training data of relevant classes.." << std::endl;

		// stores path and class data
		std::vector<imageLabelGTSRB> records;

		// stores number of read samples
		std::vector<unsigned int> nSamplesPerClass;

		// fills in the values to records and nSamplesPerClass for the specified classes
		readDataSet(records, nSamplesPerClass, CONSIDERED_CLASS_IDs);
		std::cout << "[info]\t" << records.size() << " samples in total." << std::endl;

		//Split the dataset in training and validation records
		std::cout << "[info]\tsplit the records into training and validation set" << std::endl;
		std::vector<imageLabelGTSRB> trainingRecords;
		std::vector<imageLabelGTSRB> validationRecords;

		splitDataSet(records, nSamplesPerClass, trainingRecords, validationRecords, trainPercentage);
		std::cout << "Size : " << trainingRecords.size() << " " << validationRecords.size() << std::endl;

		//Compute Features, both for validation and training
		std::cout << "[info]\tcomputing features of training images" << std::endl;
	
		computeFeaturesToMat(trainingRecords, trainingFeatures, trainingLabels);
		trainingLabels.convertTo(trainingLabels, CV_32S);

		std::cout << "[info]\tcomputing features of validation images" << std::endl;
	
		computeFeaturesToMat(validationRecords, valFeatures, valLabels);
		valLabels.convertTo(valLabels, CV_32S);

		cv::FileStorage fs(FEAT_LABELS_FILENAME, cv::FileStorage::WRITE);
		fs << "features_train" << trainingFeatures;
		fs << "labels_train" << trainingLabels;
		fs << "features_val" << valFeatures;
		fs << "labels_val" << valLabels;
		fs.release();
	}
	else {
		cv::FileStorage fs(FEAT_LABELS_FILENAME, cv::FileStorage::READ);
		std::cout << "[info]\tLoading the training features and labels" << std::endl;
		fs["features_train"] >> trainingFeatures;
		fs["labels_train"] >> trainingLabels;
		std::cout << "[info]\tLoading the validation features and labels" << std::endl;
		fs["features_val"] >> valFeatures;
		fs["labels_val"] >> valLabels;
		fs.release();
	}
	
	//Compute PCA for all the possible axes:
	bool createPCA = false;
	if (createPCA) {
		pca(trainingFeatures, cv::Mat(), CV_PCA_DATA_AS_ROW, 0);
		cv::FileStorage fs(PCA_FILENAME, cv::FileStorage::WRITE);
		pca.write(fs);
		fs.release();
	}
	else {
		cv::FileStorage fs(PCA_FILENAME, cv::FileStorage::READ);
		std::cout << "[info]\tLoading the pca configuration." << std::endl;
		pca.read(fs.root());
		fs.release();
	}

	//And project all the features in the newSpace:
	cv::Mat projectedValFeatures_RAW;
	cv::Mat projectedTrainingFeatures_RAW;
	bool calculateProjections = false;
	if (calculateProjections){
		pca.project(trainingFeatures, projectedTrainingFeatures_RAW);
		pca.project(valFeatures, projectedValFeatures_RAW);
		cv::FileStorage fs(PROJ_RAWFEATURES_FILENAME, cv::FileStorage::WRITE);
		fs << "features_train" << projectedTrainingFeatures_RAW;
		fs << "features_val" << projectedValFeatures_RAW;
		fs.release();
	}
	else {
		cv::FileStorage fs(PROJ_RAWFEATURES_FILENAME, cv::FileStorage::READ);
		std::cout << "[info]\tLoading the pca projected features" << std::endl;
		fs["features_train"] >> projectedTrainingFeatures_RAW;
		fs["features_val"] >> projectedValFeatures_RAW;
		fs.release();
	}

	//Now iterate on the number of PCA Components taken into training

	std::map<int, evalStruct> resultsOnVal_RF;
	std::map<int, evalStruct> resultsOnVal_SVM;
	std::map<int, evalStruct> resultsOnVal_ANN;

	bool lookForMaxima = false;
	if (lookForMaxima) {
		for (int NoComponents = 380; NoComponents <= 440; NoComponents = NoComponents + 5) {

			std::cout << "\n[info]\tPCA Number = " << NoComponents << std::endl;
			cv::Mat projectedTrainingFeatures;
			cv::Mat projectedValFeatures;

			cv::Rect pca_comp_train(0, 0, NoComponents, projectedTrainingFeatures_RAW.rows);
			cv::Rect pca_comp_val(0, 0, NoComponents, projectedValFeatures_RAW.rows);

			projectedTrainingFeatures = projectedTrainingFeatures_RAW(pca_comp_train).clone();
			projectedValFeatures = projectedValFeatures_RAW(pca_comp_val).clone();

			//Create Training Structure for Classifiers
			cv::Ptr<cv::ml::TrainData> tData = cv::ml::TrainData::create(projectedTrainingFeatures,
				cv::ml::SampleTypes::ROW_SAMPLE, trainingLabels);

			std::cout << "[info]\tDescriptors information : Feature Vector Size = " << projectedTrainingFeatures.size() << " \n" << std::endl;

			std::cout << "[info]\ttraining the classifiers \n" << std::endl;

			bool train = true;
			//Train the different classifiers

			//Random Forest
			std::cout << "[info]\t1st : Random Forest" << std::endl;
			auto tr_rf_start = std::chrono::system_clock::now();

			PtrRandomForest classifier_RF;
			if (train) {
				classifier_RF = RandomForest::create();

				int activeVarCount = cvFloor((NoComponents * activeVarPercentage)) + 1;

				classifier_RF->setMaxDepth(10);
				classifier_RF->setMinSampleCount(activeVarCount);
				classifier_RF->setActiveVarCount(activeVarCount);
				cv::TermCriteria termCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 300, 0.01f);

				classifier_RF->train(tData);
				classifier_RF->save(RF_FILENAME);
			}
			else cv::Algorithm::load<RandomForest>(RF_FILENAME);

			auto rf_duration = std::chrono::duration_cast<std::chrono::milliseconds>
				(std::chrono::system_clock::now() - tr_rf_start);
			std::cout << "[info]\tTraining time: " << rf_duration.count() << "ms \n" << std::endl;

			std::cout << "[info]\t1st : Random Forest -- Training Error" << std::endl;
			auto trainError_RF = evalClassifier(classifier_RF, projectedTrainingFeatures, trainingLabels); 

			//Support Vector Machine 
			std::cout << "[info]\t2nd : Support Vector Machine" << std::endl;

			auto tr_svm_start = std::chrono::system_clock::now();

			PtrSVM classifier_SVM;
			if (train) {
				classifier_SVM = SVM::create();
				classifier_SVM->setKernel(SVM::RBF);
				classifier_SVM->setType(SVM::C_SVC);
				classifier_SVM->trainAuto(tData);
			}
			else cv::Algorithm::load<SVM>(SVM_FILENAME);

			auto svm_duration = std::chrono::duration_cast<std::chrono::milliseconds>
				(std::chrono::system_clock::now() - tr_svm_start);
			std::cout << "[info]\tTraining time: " << svm_duration.count() << "ms \n" << std::endl;

			std::cout << "[info]\t2nd : Support Vector Machine -- Training Error" << std::endl;
			auto trainError_SVM = evalClassifier(classifier_SVM, projectedTrainingFeatures, trainingLabels);

			//ANN
			std::cout << "[info]\t3rd : Artificial Neural Networks" << std::endl;

			//Construct the ANN and adequate the trainingData structure to the appropiate dimension ...

			auto tr_ann_start = std::chrono::system_clock::now();

			PtrANN classifier_ANN;
			if (train) {
				cv::Mat_<int> layerSizes(1, 3);
				layerSizes(0, 0) = projectedTrainingFeatures.cols;
				layerSizes(0, 1) = CONSIDERED_CLASS_IDs.size() * 8;
				layerSizes(0, 2) = CONSIDERED_CLASS_IDs.size();

				classifier_ANN = ANN::create();
				classifier_ANN->setLayerSizes(layerSizes);
				classifier_ANN->setActivationFunction(ANN::SIGMOID_SYM, 1.0, 1.0); //Which parameters to use?
				classifier_ANN->setTrainMethod(ANN::BACKPROP, 0.00001);

				cv::Mat trainLabelsANN;
				//col_to_label;
				prepareLabelsANN(trainLabelsANN, col_to_label, trainingLabels);
				cv::Ptr<cv::ml::TrainData> tData_ANN = cv::ml::TrainData::create(projectedTrainingFeatures,
					cv::ml::SampleTypes::ROW_SAMPLE, trainLabelsANN);

				classifier_ANN->train(tData_ANN);
			}
			else cv::Algorithm::load<ANN>(ANN_FILENAME);

			auto ann_duration = std::chrono::duration_cast<std::chrono::milliseconds>
				(std::chrono::system_clock::now() - tr_ann_start);
			std::cout << "[info]\tTraining time: " << ann_duration.count() << "ms \n" << std::endl;

			std::cout << "[info]\t3rd : Artificial Neural Network -- Training Error" << std::endl;
			auto trainError_ANN = evalClassifier(classifier_ANN, projectedTrainingFeatures, trainingLabels);

			//Evaluate generalization power.

			std::cout << "[info]\tevaluation on the validation set" << std::endl;

			std::cout << "\n[info]\t 1st : Random Forest -- Evaluation Error" << std::endl;
			auto inf_rf_start = std::chrono::system_clock::now();

			auto evalError_RF = evalClassifier(classifier_RF, projectedValFeatures, valLabels);

			auto inf_rf_duration = std::chrono::duration_cast<std::chrono::milliseconds>
				(std::chrono::system_clock::now() - inf_rf_start);
			std::cout << "[info]\tTotal Inference time: " << inf_rf_duration.count() << "ms" << std::endl;
			std::cout << "[info]\tAverage Inference time: " << ((double) inf_rf_duration.count()) / valLabels.rows
				<< "ms" << std::endl;

			resultsOnVal_RF[NoComponents] = evalStruct(((double)inf_rf_duration.count()) / valLabels.rows, evalError_RF);

			std::cout << "\n[info]\t2nd : Support Vector Machine -- Evaluation Error" << std::endl;
			auto inf_svm_start = std::chrono::system_clock::now();

			auto evalError_SVM = evalClassifier(classifier_SVM, projectedValFeatures, valLabels);

			auto inf_svm_duration = std::chrono::duration_cast<std::chrono::milliseconds>
				(std::chrono::system_clock::now() - inf_svm_start);
			std::cout << "[info]\tTotal Inference time: " << inf_svm_duration.count() << "ms" << std::endl;
			std::cout << "[info]\tAverage Inference time: " << ((double)inf_svm_duration.count()) / valLabels.rows
				<< "ms" << std::endl;

			resultsOnVal_SVM[NoComponents] = evalStruct(((double)inf_svm_duration.count()) / valLabels.rows, evalError_SVM); 

			std::cout << "\n[info]\t3rd : Artificial Neural Network -- Evaluation Error" << std::endl;
			auto inf_ann_start = std::chrono::system_clock::now();

			auto evalError_ANN = evalClassifier(classifier_ANN, projectedValFeatures, valLabels);

			auto inf_ann_duration = std::chrono::duration_cast<std::chrono::milliseconds>
				(std::chrono::system_clock::now() - inf_ann_start);
			std::cout << "[info]\tTotal Inference time: " << inf_ann_duration.count() << "ms" << std::endl;
			std::cout << "[info]\tAverage Inference time: " << ((double)inf_ann_duration.count()) / valLabels.rows
				<< "ms" << std::endl;

			resultsOnVal_ANN[NoComponents] = evalStruct(((double)inf_ann_duration.count()) / valLabels.rows, evalError_ANN);

			std::cout << "ITER FINISHED!" << std::endl;
		}

		std::cout << "\nResults on RF\n " << std::endl;
		for (auto &val : resultsOnVal_RF) {
			std::cout << "NoComponents: " << val.first << "\t evalPerformance: " << val.second.evalPerformance << "\t avgTime: " << val.second.avgTime << std::endl;
		}
		
		std::cout << "\nResults on SVM\n " << std::endl;
		for (auto &val : resultsOnVal_SVM) {
			std::cout << "NoComponents: " << val.first << "\t evalPerformance: " << val.second.evalPerformance << "\t avgTime: " << val.second.avgTime << std::endl;
		}
	
		std::cout << "\nResults on ANN\n " << std::endl;
		for (auto &val : resultsOnVal_ANN) {
			std::cout << "NoComponents: " << val.first << "\t evalPerformance: " << val.second.evalPerformance << "\t avgTime: " << val.second.avgTime << std::endl;
		}
	}

	//At this point, the best configurations are selected. 
	//Now, train the best model on the complete Dataset and save the results in Harddrive.

	cv::Mat totalTrainFeat;
	cv::vconcat(projectedTrainingFeatures_RAW, projectedValFeatures_RAW, totalTrainFeat);

	cv::Mat_<int> totalTrainLabels;
	cv::vconcat(trainingLabels, valLabels, totalTrainLabels);

	std::cout << "totalTrainFeatDim = " << totalTrainFeat.size();
	std::cout << " totalTrainLabDim = " << totalTrainLabels.size() << std::endl;

	bool train_RF = false;
	PtrRandomForest classifier_RF;
	int bestNoComp_RF = 390;

	std::cout << "[info]\t1st : Random Forest" << std::endl;
	auto tr_rf_start = std::chrono::system_clock::now();
	
	if (train_RF) {

		cv::Rect pca_comp_train(0, 0, bestNoComp_RF, totalTrainFeat.rows);

		cv::Mat projectedTrainingFeatures;
		projectedTrainingFeatures = totalTrainFeat(pca_comp_train).clone();

		cv::Ptr<cv::ml::TrainData> tData = cv::ml::TrainData::create(projectedTrainingFeatures,
			cv::ml::SampleTypes::ROW_SAMPLE, totalTrainLabels);

		classifier_RF = RandomForest::create();
		
		int activeVarCount = cvFloor((bestNoComp_RF * activeVarPercentage)) + 1;

		classifier_RF->setMaxDepth(10);
		classifier_RF->setMinSampleCount(activeVarCount);
		classifier_RF->setActiveVarCount(activeVarCount);
		cv::TermCriteria termCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 300, 0.01f);

		classifier_RF->train(tData);
		classifier_RF->save("ClassifRF.json");

		std::cout << "[info]\t1st : Random Forest -- Training Error" << std::endl;
		auto trainError_RF = evalClassifier(classifier_RF, projectedTrainingFeatures, totalTrainLabels);
	}
	else {
		classifier_RF = cv::Algorithm::load<RandomForest>("ClassifRF.json");
	}

	auto rf_duration = std::chrono::duration_cast<std::chrono::milliseconds>
		(std::chrono::system_clock::now() - tr_rf_start);
	std::cout << "[info]\tTraining time: " << rf_duration.count() << "ms \n" << std::endl;

	bool train_SVM = false;
	PtrSVM classifier_SVM;
	int bestNoComp_SVM = 1300;
	
	std::cout << "[info]\t2nd : Support Vector Machine" << std::endl;
	auto tr_svm_start = std::chrono::system_clock::now();

	if (train_SVM) {

		cv::Rect pca_comp_train(0, 0, bestNoComp_SVM, totalTrainFeat.rows);

		cv::Mat projectedTrainingFeatures;
		projectedTrainingFeatures = totalTrainFeat(pca_comp_train).clone();

		cv::Ptr<cv::ml::TrainData> tData = cv::ml::TrainData::create(projectedTrainingFeatures,
			cv::ml::SampleTypes::ROW_SAMPLE, totalTrainLabels);

		classifier_SVM = SVM::create();
		classifier_SVM->setKernel(SVM::RBF);
		classifier_SVM->setType(SVM::C_SVC);
		classifier_SVM->trainAuto(tData);

		classifier_SVM->save("ClassifSVM.json");

		std::cout << "[info]\t2nd : Support Vector Machine -- Training Error" << std::endl;
		auto trainError_SVM = evalClassifier(classifier_SVM, projectedTrainingFeatures, totalTrainLabels);
	}
	else {
		classifier_SVM = cv::Algorithm::load<SVM>("ClassifSVM.json");
	}

	auto svm_duration = std::chrono::duration_cast<std::chrono::milliseconds>
		(std::chrono::system_clock::now() - tr_svm_start);
	std::cout << "[info]\tTraining time: " << svm_duration.count() << "ms \n" << std::endl;

	bool train_ANN = true;
	PtrANN classifier_ANN;
	int bestNoComp_ANN = 380;

	std::cout << "[info]\t3rd : Artificial Neural Networks" << std::endl;
	//Construct the ANN and adequate the trainingData structure to the appropiate dimension ...
	auto tr_ann_start = std::chrono::system_clock::now();

	if (train_ANN) {

		cv::Rect pca_comp_train(0, 0, bestNoComp_ANN, totalTrainFeat.rows);

		cv::Mat projectedTrainingFeatures;
		projectedTrainingFeatures = totalTrainFeat(pca_comp_train).clone();

		cv::Mat_<int> layerSizes(1, 3);
		layerSizes(0, 0) = projectedTrainingFeatures.cols;
		layerSizes(0, 1) = CONSIDERED_CLASS_IDs.size() * 8;
		layerSizes(0, 2) = CONSIDERED_CLASS_IDs.size();

		classifier_ANN = ANN::create();
		classifier_ANN->setLayerSizes(layerSizes);
		classifier_ANN->setActivationFunction(ANN::SIGMOID_SYM, 1.0, 1.0); //Which parameters to use?
		classifier_ANN->setTrainMethod(ANN::BACKPROP, 0.00001);

		cv::Mat trainLabelsANN;
		//col_to_label;
		prepareLabelsANN(trainLabelsANN, col_to_label, totalTrainLabels);

		cv::Ptr<cv::ml::TrainData> tData_ANN = cv::ml::TrainData::create(projectedTrainingFeatures,
			cv::ml::SampleTypes::ROW_SAMPLE, trainLabelsANN);

		classifier_ANN->train(tData_ANN);
		classifier_ANN->save("ClassifANN.json");

		std::cout << "[info]\t3rd : Artificial Neural Network -- Training Error" << std::endl;
		auto trainError_ANN = evalClassifier(classifier_ANN, projectedTrainingFeatures, totalTrainLabels);
	}
	else {
		classifier_ANN = cv::Algorithm::load<ANN>("ClassifANN.json");
	}

	for (auto& col : col_to_label) {
		std::cout << col.first << col.second << std::endl;
	}

	auto ann_duration = std::chrono::duration_cast<std::chrono::milliseconds>
		(std::chrono::system_clock::now() - tr_ann_start);
	std::cout << "[info]\tTraining time: " << ann_duration.count() << "ms \n" << std::endl;

	//TestSet:
	cv::Mat testFeatures;
	cv::Mat_<int>  testLabels;

	bool constructTestSet = false;
	if (constructTestSet) {

		std::ifstream testGtFile;
		testGtFile.open(GT_TEST);

		while (testGtFile.good())
		{
			// read record(whole line)
			std::string line;
			std::getline(testGtFile, line);

			if (line.size() == 0) continue;

			std::vector<std::string> splitted;
			split(line, ';', splitted);

			int classID = std::stoi(splitted.at(splitted.size() - 1));
			auto foundIndex = std::find(CONSIDERED_CLASS_IDs.begin(), CONSIDERED_CLASS_IDs.end(), classID);

			if (foundIndex == CONSIDERED_CLASS_IDs.end()) continue;

			cv::Mat img = cv::imread(IMAGE_DIR_TEST + splitted.at(0));

			// resize image and compute features
			std::vector<float> descriptors;
			int weight = 1;
			for (auto& hog_d : { MNIST_HOG_1, MNIST_HOG_2, MNIST_HOG_3 }) {
				std::vector<float> descriptors_aux;
				computeFeatures(img, descriptors_aux, hog_d);
				std::transform(descriptors_aux.begin(), descriptors_aux.end(), descriptors_aux.begin(),
					[weight](float d) -> float { return d * weight; });
				descriptors.insert(descriptors.end(), descriptors_aux.begin(), descriptors_aux.end());
				weight *= 2;
			}

			//Look for the HOG2 preprocesed features
			std::string HOG2_name = HOG2_DIR_TEST + splitted.at(0);
			HOG2_name = HOG2_name.substr(0, HOG2_name.size() - 4) + ".txt";

			std::ifstream hog2_file(HOG2_name);
			while (hog2_file.good()) {
				std::string tmp;
				std::getline(hog2_file, tmp);
				descriptors.push_back(std::stof(tmp));
			}
			hog2_file.close();

			testFeatures.push_back(descriptors);

			// set labels
			testLabels.push_back(classID);
		}
		testGtFile.close();

		testFeatures = testFeatures.reshape(0, (testFeatures.cols * testFeatures.rows) / 4400);
		testLabels = testLabels.reshape(0, (testLabels.cols * testLabels.rows));

		pca.project(testFeatures, testFeatures);

		cv::FileStorage fs(TEST_RAWFEATURES_FILENAME, cv::FileStorage::WRITE);
		fs << "features_test" << testFeatures;
		fs << "labels_test" << testLabels;
		fs.release();
	}
	else {
		cv::FileStorage fs(TEST_RAWFEATURES_FILENAME, cv::FileStorage::READ);
		fs["features_test"] >> testFeatures;
		fs["labels_test"] >> testLabels;
		fs.release();
	}

	cv::Rect pca_comp_rf(0, 0, bestNoComp_RF, totalTrainFeat.rows);
	cv::Mat testRF = testFeatures(pca_comp_rf).clone();

	std::cout << "\n[info]\t 1st : Random Forest -- Evaluation Error" << std::endl;
	auto inf_rf_start = std::chrono::system_clock::now();

	auto evalError_RF = evalClassifier(classifier_RF, testRF, testLabels);

	auto inf_rf_duration = std::chrono::duration_cast<std::chrono::milliseconds>
		(std::chrono::system_clock::now() - inf_rf_start);
	std::cout << "[info]\tTotal Inference time: " << inf_rf_duration.count() << "ms" << std::endl;
	std::cout << "[info]\tAverage Inference time: " << ((double)inf_rf_duration.count()) / valLabels.rows
		<< "ms" << std::endl;

	cv::Rect pca_comp_svm(0, 0, bestNoComp_SVM, totalTrainFeat.rows);
	cv::Mat testSVM = testFeatures(pca_comp_svm).clone();

	std::cout << "\n[info]\t 2nd : Support Vector Machine -- Evaluation Error" << std::endl;
	auto inf_svm_start = std::chrono::system_clock::now();

	auto evalError_SVM = evalClassifier(classifier_SVM, testSVM, testLabels);

	auto inf_svm_duration = std::chrono::duration_cast<std::chrono::milliseconds>
		(std::chrono::system_clock::now() - inf_svm_start);
	std::cout << "[info]\tTotal Inference time: " << inf_rf_duration.count() << "ms" << std::endl;
	std::cout << "[info]\tAverage Inference time: " << ((double)inf_rf_duration.count()) / valLabels.rows
		<< "ms" << std::endl;

	cv::Rect pca_comp_ann(0, 0, bestNoComp_ANN, totalTrainFeat.rows);
	cv::Mat testANN = testFeatures(pca_comp_ann).clone();
	
	std::cout << "\n[info]\t3rd : Artificial Neural Network -- Evaluation Error" << std::endl;
	auto inf_ann_start = std::chrono::system_clock::now();

	for (auto& col : col_to_label) {
		std::cout << col.first << col.second << std::endl;
	}

	auto evalError_ANN = evalClassifier(classifier_ANN, testANN, testLabels);

	auto inf_ann_duration = std::chrono::duration_cast<std::chrono::milliseconds>
		(std::chrono::system_clock::now() - inf_ann_start);
	std::cout << "[info]\tTotal Inference time: " << inf_ann_duration.count() << "ms" << std::endl;
	std::cout << "[info]\tAverage Inference time: " << ((double)inf_ann_duration.count()) / valLabels.rows
		<< "ms" << std::endl;

	

	std::cout << "\n[info]\tfinished program! goodbye. :)" << std::endl;
	 
	system("pause");

	return 0;
}

