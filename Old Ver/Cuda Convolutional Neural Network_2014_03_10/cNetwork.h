

#ifndef _NETWORK_H
#define _NETWORK_H

#include "windows.h"
#include "cNetColumn.h"

class cColParameter{
public:

	void*		Network;
	cImages*	Input;
	int			label;
	int			number;		// the number this column in the whole network
	HANDLE		Event;
	cPixel*		Output;
	cImages*	Target;
};

class cNetwork{
public:

	HANDLE*	ColEvents;

	thrust::host_vector<cColParameter> ColParam;

	thrust::host_vector<cNetColumn*> NetColumn;

	cImages InputImage;
	cImages TargetImage;

	void Initialize();

	// Training using one image
	bool Train(cImages* Input, int label, int Count);

	// Training using the whole set
	void TrainSet();

	// Test one input image
	int Compute(cImages* Input);

	int GetResult();
	
	cPixel* Compute(cImages* Input, int col);

	// Test whole set
	void ComputeSet();

	void CalculateScale();

	void Trace();

	void TraceDevOutput();

	void Compute();

	void SaveWeights();
	void LoadWeights();

	void CopyWeightsD2H();

	// Clean up
	~cNetwork();

};

#endif