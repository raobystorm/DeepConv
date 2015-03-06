

#ifndef GLOBAL_VALUES
#define GLOBAL_VALUES

#include "thrust\host_vector.h"
#include "thrust\device_vector.h"
#include <stdio.h>
#include <string.h>
#include <nppdefs.h>
#include "math.h"

#define WEIGHTS_SAVE_LOAD_PATH				"D:\\MNIST\\weights.dat"
#define TRAINING_SET_IMAGE_PATH				"D:\\MNIST\\train_images.dat"
#define TRAINING_SET_LABLE_PATH				"D:\\MNIST\\train_labels.dat"
#define TESTING_SET_IMAGE_PATH				"D:\\MNIST\\test_images.dat"
#define TESTING_SET_LABLE_PATH				"D:\\MNIST\\test_labels.dat"
#define IMAGE_SAVE_PATH						"D:\\MNIST\\save_image"

#define LOG_FILE_PATH						"D:\\MNIST\\Log.txt"
#define LOG_FILE_BATCH_PATH					"D:\\MNIST\\Log"
#define LOG_FILE_TEMP						"D:\\MNIST\\LogTemp.txt"

#define OUTPUT_SAVE_PATH					"D:\\MNIST\\Output"
#define	INPUT_SAVE_PATH						"D:\\MNIST\\Input"

#define IMAGE_8U_C1										// define the image tag to identify the image type
//#define IMAGE_32F_C1

#define cImages											npp::ImageCPU_8u_C1	
#define cImagesGPU									npp::ImageNPP_8u_C1
#define cPixel												Npp8u
#define DevImg											thrust::device_ptr<cPixel*>
#define DevWeights										thrust::device_ptr<float*>

#define BigtoLittle32(A)   ((( (int)(A) & 0xff000000) >> 24) | \
                                       (( (int)(A) & 0x00ff0000) >> 8)   | \
                                       (( (int)(A) & 0x0000ff00) << 8)   | \
                                       (( (int)(A) & 0x000000ff) << 24))

#define PI 3.1415926535

extern inline void StopErrorMsg(const char*);

class cGlobalValues{
public:

	unsigned short sNetColumnCount;
	unsigned short sLayerCount;
	unsigned short sConvLayerCount;
	unsigned short sPoolLayerCount;
	unsigned short sFullConnLayerCount;
	unsigned short sFullConnNeuronCount;
	unsigned short sOutputCount;
	unsigned short sBatchSize;


	// the filter-step of the convolution layers
	// used to re-calc the number of layers in the model
	// Only contain the anchors for convolution level
	thrust::host_vector<NppiPoint> Anchors;

	// For convolution and pooling layers
	// kerSize stand for the convolution kernel size and fan in windows size of maxpooling
	thrust::host_vector<NppiSize> kerSizes; 

	// The size of each layer's input
	// Calculate automatically in initialization of global values
	thrust::host_vector<NppiSize> MapSizes;

	thrust::host_vector<short> FilterMultipliers;
	thrust::host_vector<short> FilterCounts;
	// We group the filter maps which generated from the convolution of the same
	// Filter map in the lower level fo the model into one group.
	// this vector is counting for the amount of groups in each level.
	// Just for convolution layers and pooling layers.

	// Is weights of network float or integer?
	bool bFloatWeight;

	// Copy the results to CPU memory or directly compute in GPU?
	bool bUseCPUMem;

	// Is weights of filter need to log in tracing network?
	bool bTraceFilter;
	bool bTraceOutput;

	// single: Using multi-column to learning the same single image
	enum eTrainMode { SINGLE};
	enum eTrainMode TrainMode;

	enum eNeuronType {RELU, WTA, TANH};
	enum eNeuronType NeuronType;

	enum eLayerType {CONVOLUTION, MAXPOOLING,
		FULLYCONNECTED, OUTPUT, NORMALIZATION};

	thrust::host_vector<eLayerType> LayerTypes;

	float fTrainStopThres;
	float fLearnStep;
	float fLearnMomentum;
	float fWeightDecay;

	float fMinRandWeight;
	float fMaxRandWeight;

	int nMinRandWeight;
	int nMaxRandWeight;

	int nTrainBatchSize;

	void Initialize();
	void CalculateScale();
	void show();

	cudaStream_t H2DStream;
	cudaStream_t D2HStream;
	cudaStream_t D2DStream;
	cudaStream_t CalcStream;
};

extern cGlobalValues GlobalValues;

extern inline float		fRandom(float, float);
extern inline int		nRandom(int, int);
extern inline float		fRanSigmoid(float a, float b, float c);
extern inline float		fRanGauss(float a, float b, float avg, float sig);

// the count of training & testing set images
// the size of input raw images.
 extern int trainCount,
	testCount,
	imgWidth,
	imgHeight,
	magicNum;

extern FILE* weightsFile;
extern FILE* tstLabel;
extern FILE* tstSet;
extern FILE* configFile;
extern FILE* imgSet;
extern FILE* logFile;
extern FILE* imgLabel;

#endif