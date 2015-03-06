
#include "ImageIO.h"

#include "cLoadFile.h"

#include "thrust\host_vector.h"
#include "thrust\device_vector.h"
#include "helper_cuda.h"

class cFilter{
public:
	
	unsigned short sLevel;
	unsigned short sInputCount;

	enum cGlobalValues::eLayerType Type;

	size_t Pitch;

	// For non pooling layer kerSize is the size of kernel weights
	// MapSize is the size of offsets. For max pooling kerSize is the
	// size of input map and MapSize is unused.
	NppiSize kerSize;
	NppiSize MapSize;
	NppiSize InSize;

	// The weights of the filters
	// each of float and integer weights are optional for convolution and MLP layers 
	// bool weights used for recording the max pooling neurons
	// which in order to trading for performance, also could be NULL to save space
	Npp32f* fWeights;
	Npp32s* nWeights;
	bool* bWeights;

	Npp32f* fDevWeights;
	Npp32s* nDevWeights;
	bool* bDevWeights;

	void Initialize(short);

	void Compute();

	void Train();

	void CopyWeightsH2D();

	void CopyWeightsD2H();

	void Trace(short);

	~cFilter();

};