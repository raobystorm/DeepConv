
// this class is the group of filters -- in the model the weights is actually a 4-D tensor
// two demensions are the location x,y in the input, one is the level of the layer and
// the forth is used to identify which weights map this weight is applies to.
// for the same input map's filters we group them into one filter group.

#include "cFilter.h"

class cFilterGroup{
private:
	// Private member for temporary space using
	// Float type to insure the accuracy
	DevImg  DevInPtr;
	DevImg  DevOutPtr;
	cImagesGPU*		DevInTemp;
	float*			fDevOffsets;
	float*			fOffsets;
	float*			fDevTemp;			// In convolution layers temp block for kernels, need to be stored
	float*			fDeltaTemp;			// In convolution layers temp value for delta, need to be stored

public:

	cudaStream_t* H2DStream;
	cudaStream_t* D2HStream;
	cudaStream_t* D2DStream;
	cudaStream_t* CalcStream;

	enum cGlobalValues::eLayerType Type;

	// How many filer maps in this group
	unsigned int nFilterCount;
	
	// Layer level of this group
	unsigned short sLevel;

	// Counting for inputs when used all-in-one dev mem
	unsigned short sInputCount;

	// This integer stand for the number of each input
	// produces the output by the same number of filters.
	unsigned short sFilterMultiplier;

	thrust::host_vector<cImages*> Input;
	thrust::host_vector<cImages*> Output;
	// This vector in cpu is saving the temp output coming from GPU
	// It's convenient for synchronizing the results between groups
	// Using aschronizing copying and calculation to improve the performance
	thrust::host_vector<cImagesGPU*> DevOutput;
	thrust::host_vector<cImagesGPU*> DevInput;

	// Delta value used in back propogation
	float* fDevDelta;

	// For pooling, fully-connected and output layer, using one big
	// buffer to hold input and output to improve performance
	float* DevIn;
	float* DevOut;

	// pointer to each 2-D weights
	thrust::host_vector<cFilter*> Filters;

	// the size of each filter map
	// Only for convolution layer
	NppiSize kerSize;
	NppiPoint Anchor;

	// The size of one image input of this group
	NppiSize MapSize;

	NppiSize OutSize;
	NppiSize InSize;
	
	// Using for concurrency of the program
	cudaEvent_t EventH2D, EventD2H, EventCalc, EventD2D;
	
	// parameter indicates the level
	void Initialize(short);

	// parameter indicates the number of this group in the layer
	void Compute(short);

	void Train(int);

	// parameter indicates the number of this instance
	void Trace(short);

	void Trace(float*, NppiSize);

	void TraceDelta(FILE*);

	void TraceWideWeights();
	
	void CalcDelta( float*, NppiSize);

	// I/O operations between host and devices
	// Copy functions are asynchronized
	void CopyResults();

	void CopyInputs();

	void SetDeviceInput(cImagesGPU*);

	void CopyWeights();
	void SaveWeights();
	void LoadWeights();
	
	void UpdateConvoWeights();

	void Clean();

	void DeviceAlloc();

	void ExtendInput();

	void ExtendInput1D();

	void SetInputPointer();

	void SetTarget(cImagesGPU* Target);

	void InitializeOffsets();

	void SetOutputPointer();

	void DevOutputAlloc();

	void DevOutAlloc();

	void ClearGPUOutput();

	void ClearGPUInput();

	void CopyDevInput(cImagesGPU*, npp::Image::Size);

	void CopyDevInput1D(cImagesGPU*, npp::Image::Size);

	void CopyWeightsH2D();

	void CopyWeightsD2H();

	void TraceDevInput();

	void TraceDevOutput(int);

	~cFilterGroup();

	float*			fDevConKernel;
};