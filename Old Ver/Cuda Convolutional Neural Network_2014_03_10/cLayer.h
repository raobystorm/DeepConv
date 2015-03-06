

#if !defined _LAYER_H
#define _LAYER_H

#include <stdlib.h>
#include "cFilterGroup.h"

class cLayer{
public:
	
	cudaStream_t* H2DStream;
	cudaStream_t* D2HStream;
	cudaStream_t* D2DStream;
	cudaStream_t* CalcStream;

	// Counting for groups
	short nFilterGroupCount;
	// Counting for filters per group
	short nFilterCount;
	// group weights into groups
	cFilterGroup* FilterGroup;

	// Anchor is the stride between two neighbour filters
	// which located in different rows and columns in the image
	// Including the shift of X and Y axis.
	// Only used for convolution layers.
	NppiPoint Anchor;

	// kerSize stands for kernel window size of conv layer
	// and fan in window size of max pooling layer
	NppiSize kerSize;
	
	NppiSize OutSize, InSize;

	// here are the datas for compute
	thrust::host_vector<cImages*> Input;
	thrust::host_vector<cImages*> Output;

	// Inputs and outputs on devices, used for GPU computing
	// This is now used for holding the image type value
	thrust::host_vector<cImagesGPU*> DevInput;
	thrust::host_vector<cImagesGPU*> DevOutput;

	// By the limitation of the kernel function calls, using the 
	// big combined all-in-one buffer to hold the input and output for gpu computation
	float* DevIn;
	float* DevOut;

	// Check if it is the first time to run compute function or not
	bool bFirstCom;

	// Is the input and output of device memory combined to accelerate
	// the computation or not?
	bool bCombineInput;
	bool bCombineOutput;

	enum cGlobalValues::eLayerType Type;

	cLayer* HigherLayer;
	cLayer* LowerLayer;

	// this layer is in which level of the model
	unsigned int nLayerLevel;

	cudaEvent_t EventBusy;

	void Train();
	// Device memory usage check and allocation
	void CheckDeviceMemory();

	void AllocDevIn();
	// 
	void Compute();
	void SetTarget(cImagesGPU*);
	void CalcDelta();
	void Initialize(short);
	void Trace();
	void CopyWeightsD2H();
	void SaveWeights();
	void LoadWeights();
	void ClearGPUOutput();
	void TraceDevOutput(int);
	void TraceDelta();
	~cLayer();

};

#endif