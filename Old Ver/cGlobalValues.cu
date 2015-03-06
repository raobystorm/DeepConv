
#include "cGlobalValues.h"
#include "time.h"

cGlobalValues GlobalValues;

// default parameters of the model here
void cGlobalValues::Initialize(){

	short TempShort;
	NppiPoint TempPoint;
	NppiSize TempSize;

	bFloatWeight = true;
	bTraceFilter = true;

	bUseCPUMem = true;

	TrainMode = SINGLE;

	sNetColumnCount = 1;
	sLayerCount = 6;
	sConvLayerCount = 2;
	sPoolLayerCount = 2;
	sFullConnLayerCount = 1;
	sFullConnNeuronCount = 160;
	sOutputCount = 10;

	sBatchSize = 128;
	
	LayerTypes = thrust::host_vector<eLayerType>();

	LayerTypes.push_back(cGlobalValues::CONVOLUTION);
	LayerTypes.push_back(cGlobalValues::MAXPOOLING);
	LayerTypes.push_back(cGlobalValues::CONVOLUTION);
	LayerTypes.push_back(cGlobalValues::MAXPOOLING);
	LayerTypes.push_back(cGlobalValues::FULLYCONNECTED);
	LayerTypes.push_back(cGlobalValues::OUTPUT);

	fTrainStopThres = 0.001f;
	fLearnStep = 0.05f;

	fLearnMomentum = 1.f;

	fWeightDecay = 0.0005f;

	fMinRandWeight = -0.05;
	fMaxRandWeight = 0.05;

	nMinRandWeight = 0;
	nMaxRandWeight = 255;

	NeuronType = cGlobalValues::eNeuronType::RELU;

	FilterMultipliers.push_back(20);
	FilterMultipliers.push_back(1);
	FilterMultipliers.push_back(40);
	FilterMultipliers.push_back(1);
	FilterMultipliers.push_back(sFullConnNeuronCount);
	FilterMultipliers.push_back(sOutputCount);

	int m = 1;
	for(int i = 0; i < sLayerCount; i++){
	
		if(LayerTypes[i] == cGlobalValues::CONVOLUTION){ 

			m *= FilterMultipliers[i];
			FilterCounts.push_back(m);
		}
		else if(LayerTypes[i] == cGlobalValues::MAXPOOLING) FilterCounts.push_back(m);
		else if(LayerTypes[i] == cGlobalValues::FULLYCONNECTED) FilterCounts.push_back(sFullConnNeuronCount);
		else if(LayerTypes[i] == cGlobalValues::OUTPUT) FilterCounts.push_back(sOutputCount);

	}

	// Anchor definition
	TempPoint.x = 1;
	TempPoint.y = 1;

	Anchors.push_back(TempPoint);
	Anchors.push_back(TempPoint);

	// Kernel size definition
	// Convolution window kernel size
	TempSize.height = 5;
	TempSize.width = 5;
	kerSizes.push_back(TempSize);

	// Max pooling window size
	TempSize.height = 2;
	TempSize.width = 2;
	kerSizes.push_back(TempSize);
	
	TempSize.height = 4;
	TempSize.width = 4;
	kerSizes.push_back(TempSize);

	TempSize.height = 3;
	TempSize.width = 3;
	kerSizes.push_back(TempSize);

	// Count for groups in each level, according to filter count compute automatically
	// First layer group is always 1
	/*FilterGroupCounts.push_back(1);
	TempShort = 1;

	for(int i = 1; i < sConvLayerCount + sPoolLayerCount; i++){
	
		if(i%2 == 0) {
			
			FilterGroupCounts.push_back(TempShort);
		}
		else {

			FilterGroupCounts.push_back(1);
			TempShort *= FilterCounts[i-1];
		}
	}

	// Group number of fully connected layer and output layer is 1
	for(int i = 0; i < sFullConnLayerCount; i++){
	
		FilterGroupCounts.push_back(1);
	}

	FilterGroupCounts.push_back(1);*/

	// Map sizes compute

	TempSize.height = imgHeight;
	TempSize.width = imgWidth;
	for(int i = 0; i < sConvLayerCount + sPoolLayerCount; i++){
	
		MapSizes.push_back(TempSize);
		if(i % 2 == 0){
		
			TempSize.height = (TempSize.height - kerSizes[i].height + 1)/(Anchors[i/2].y);
			TempSize.width = (TempSize.width - kerSizes[i].width + 1)/(Anchors[i/2].x);
		}else{
		
			TempSize.height /= kerSizes[i].height;
			TempSize.width /= kerSizes[i].width;
		}
	}
	// this is for the first fully conn layer to calc the input scale
	MapSizes.push_back(TempSize);

	cudaStreamCreate(&H2DStream);
	cudaStreamCreate(&D2HStream);
	cudaStreamCreate(&CalcStream);
	cudaStreamCreate(&D2DStream);
	
	srand((int)time(0));
}

inline float fRandom(float a, float b){

	return rand() / (RAND_MAX + 1.0) * (b - a) + a;
}

inline int nRandom(int Min, int Max){
	
	return rand() * Max / RAND_MAX ;
}

// Random sample from a gaussian distribution
// u is the mean and d is the variance. x sampled randomly from a to b
inline float fRanSigmoid(float a, float b, float c){

	float x = fRandom(a, b);
	return 1.0f/(1.0f + expf(-1.0f * c * x));
}

inline void StopErrorMsg(const char* msg){

	printf("\n");
	fprintf(logFile, "\n");
	printf(msg);
	fprintf(logFile, "\n");
	printf("\n");

	system("pause");
}

// Random 1D gauss from close interval a to b
inline float fRanGauss(float a, float b, float avg, float sig){

	float x = fRandom(a, b);
	return 1.f/(sig * sqrt(2.f*PI)) * expf(-0.5f * ((x - avg)*(x - avg)) / (sig*sig));
}

void cGlobalValues::show(){

	printf("Global values current:\n\n");
	printf("sNetColumnCount: %d\n", sNetColumnCount);
	printf("sLayerCount: %d\n", sLayerCount);
	printf("sConvLayerCount: %d\n", sConvLayerCount);
	printf("sPoolLayerCount: %d\n", sPoolLayerCount);
	printf("sFullConnLayerCount: %d\n", sFullConnLayerCount);
	printf("sFullConnNeuronCount: %d\n", sFullConnNeuronCount);
	printf("sOutputCount: %d\n\n", sOutputCount);

	printf("fTrainStopThres: %f\n", fTrainStopThres);
	printf("fLearnStep: %f\n\n", fLearnStep);

	printf("Anchors:\n");
	for(int i =0 ;i < Anchors.size();i++){
	
		printf("x%d: %d y%d:%d\n", i, Anchors[i].x, i, Anchors[i].y);
	}
	printf("\n");

	printf("kerSizes:\n");
	for(int i =0 ;i < kerSizes.size();i++){
	
		printf("height%d: %d width%d:%d\n", i, kerSizes[i].height, i, kerSizes[i].width);
	}
	printf("\n");

	printf("MapSizes:\n");
	for(int i =0 ;i < MapSizes.size();i++){
	
		printf("height%d: %d width%d:%d\n", i, MapSizes[i].height, i, MapSizes[i].width);
	}
	printf("\n");

	printf("FilterCounts:\n");
	for(int i =0 ;i < FilterCounts.size();i++){
	
		printf("%d: %d\n", i, FilterCounts[i]);
	}
	printf("\n");

	printf("FilterMultipliers:\n");
	for(int i =0 ;i < FilterMultipliers.size();i++){
	
		printf("%d: %d\n", i, FilterMultipliers[i]);
	}
	printf("\n");

	printf("Finish\n");
	getchar();
}