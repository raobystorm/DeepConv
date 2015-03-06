
#ifndef _NET_COLUMN_H
#define _NET_COLUMN_H

#include "cLayer.h"
#include "cLoadFile.h"

class cNetColumn{
public:
	
	thrust::host_vector<cLayer*> Layers;

	// Counting for filters per group in each layer.
	// this is not the total amount of the whole layer, instead it's the number of
	// how many filters in one group. Knowing these numbers we can calculate
	// the numbers of filter groups and total filter amount out.
	thrust::host_vector<short> FilterCounts;
	cImages* InputImage;
	cImagesGPU*		DevInputImage;
	// When training this is the right output of input
	cImages* TargetImage;

	unsigned int nLayerCount;
	unsigned int nConvLayerCount;
	unsigned int nPoolLayerCount;
	unsigned int nFullLayerCount;
	unsigned int nFullNeuronCount;

	cudaStream_t H2DStream;
	cudaStream_t D2HStream;
	cudaStream_t D2DStream;
	cudaStream_t CalcStream;

	void Initialize();

	void Compute();

	cPixel* ComputeImage(cImages*);

	void SaveWeights();

	void LoadWeights();

	cPixel* GetResult();

	void Train(cImages*);

	void Trace(short);

	void TraceDevOutput(int);

	void TraceMaxPooling();

	void Clean();

	void CopyWeightsD2H();

	~cNetColumn();

};

#endif