  
#include "cFilterGroup.h"
#include "curand.h"
#include "curand_kernel.h"

inline __global__ void ConvoKernel(
	Npp8u*				fDevSrc,
	Npp8u*				fDevDst,
	float*				fDevKernel,
	NppiSize			oSize,			// Window size of the HIGHER level
	NppiSize			kerSize,
	int					nTotalCount
){
	
	int Idx_d = blockIdx.x * blockDim.x + threadIdx.x;

	if(Idx_d >= oSize.width * oSize.height * nTotalCount) return;

	int Idx_dy = Idx_d / (oSize.width * nTotalCount);
	int Idx_dx = Idx_d % (oSize.width * nTotalCount);

	int nNumber = Idx_dx / oSize.width;

	Idx_dx = Idx_dx % oSize.width;

	int Idx_kerBase = kerSize.width * kerSize.height * nNumber;

	int srcWidth = nTotalCount*(oSize.width + kerSize.width - 1);

	float sum = 0.0f;

	for(int i = 0; i < kerSize.height; i++){
		for(int j = 0; j < kerSize.width; j++){

			sum += fDevSrc[(i+Idx_dy)*srcWidth+nNumber*(oSize.width+kerSize.width-1)+j+Idx_dx]*
				fDevKernel[Idx_kerBase+i*kerSize.width+j];
		}
	}

	if(sum < 255.f)	fDevDst[Idx_d] = sum;
	else fDevDst[Idx_d] = 255;
}

inline __global__ void ProdKernel(
	float*				fDevKernel,
	int					nCount
){
	int Idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(Idx >= nCount) return;

	fDevKernel[Idx] = 0.15f;
}

inline __global__ void UpdateWeightsConvo(
	float*				fDevWeights,
	float*				fDevWeightsDelta,
	float				fWeightDecay,
	float				fLearnRate,
	float				fLearnMomentum,
	int					nCount
){
	int Idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(Idx >= nCount) return;

	fDevWeights[Idx] -= fWeightDecay*fLearnRate*fDevWeights[Idx] + fLearnMomentum*fLearnRate*fDevWeightsDelta[Idx];
}

// Update the weights values of the convolution layers
inline __global__ void UpdateWeightsDeltaConvo(
	cPixel*				fDevSrc,
	float*				fDevDelta,
	float*				fDevWeightsDelta,
	NppiSize			kerSize,
	NppiSize			oSize,				// The window size of higher layer
	int					nCount
){
	int Idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(Idx >= kerSize.width * kerSize.height * nCount) return;

	// Several weight matrix here, Idx_k is the local coord in this matrix
	int Idx_k = Idx % (kerSize.width * kerSize.height);
	int nNumber = Idx / (kerSize.width * kerSize.height);

	int Idx_kx = Idx_k % kerSize.width;
	int Idx_ky = Idx_k / kerSize.width;

	int srcWidth = oSize.width + kerSize.width - 1;

	float sum = 0.0;

	for(int i = 0; i < oSize.height; i++){

		for(int j = 0; j < oSize.width; j++){
		
			// Simply add up the corresponding patch of W(i,j) here
			sum += fDevDelta[i*(nCount*oSize.width)+nNumber*oSize.width+j]*
				(fDevSrc[(Idx_ky+i)*(nCount*srcWidth)+nNumber*srcWidth+Idx_kx+j]/255.0f);
		}
	}

	fDevWeightsDelta[Idx] = /*fWeightDecay*fLearnRate*fDevWeights[Idx] + fLearnMomentum*fLearnRate**/sum;
}

// Combining the delta values of a convolution layer
// Convolution produce several outputs for one input
// When doing backpropogation, combining the delta values
// for the layers below this convo layer.
inline __global__ void CombineDeltaConvo(
	cPixel*				fDevSrc,
	float*				fDevDelta,
	float*				fDevDeltaLower,
	NppiSize			LowerSize,			// Total size of the lower layer
	NppiSize			HigherSize,
	int					nFilterMultiplier	
){
	int Idx_s = blockDim.x * blockIdx.x + threadIdx.x;

	if(Idx_s >= LowerSize.width * LowerSize.height) return;

	if(fDevSrc[Idx_s] == 0){
	
		fDevDeltaLower[Idx_s] = 0.0f;
		return;
	}

	int Idx_sx = Idx_s % LowerSize.width;
	int Idx_sy = Idx_s / LowerSize.width;

	float sum = 0.0f;

	for(int i = 0; i < nFilterMultiplier; i++){
	
		sum += fDevDelta[Idx_sy*(HigherSize.width) + i*(LowerSize.width) + Idx_sx];
	}

	fDevDeltaLower[Idx_s] = sum;
}

// Propgating the delta using rotated kernels and convolution computing
// One thread for delta of one pixel in the lower layer
inline __global__ void PropDeltaConvo(
	cPixel*				fDevSrc,
	float*				fDevDeltaLower,
	float*				fDevDelta,
	float*				fDevKernel,
	NppiSize			WinSize,			// Window size of the lower level
	NppiSize			kerSize,
	int					nTotalCount
){
	
	int Idx_s = blockIdx.x * blockDim.x + threadIdx.x;

	if(Idx_s >= WinSize.width * WinSize.height * nTotalCount) return;

	if(fDevSrc[Idx_s] == 0) {

		fDevDeltaLower[Idx_s] = 0.0f;
		return;
	}

	int Idx_sy = Idx_s / (WinSize.width * nTotalCount);
	int Idx_sx = Idx_s % (WinSize.width * nTotalCount);

	int nNumber = Idx_sx / WinSize.width;

	Idx_sx = Idx_sx % WinSize.width;

	int Idx_kerBase = kerSize.width * kerSize.height * nNumber;

	int dstWidth = nTotalCount*(WinSize.width - kerSize.width + 1);

	// When convolve the rotated kernel, kerOffset is the local coodinate
	// base of the kernel matrix in the higher delta layer
	int kerOffsetX = Idx_sx-kerSize.width+1;
	int kerOffsetY = Idx_sy-kerSize.height+1;

	float sum = 0;

	for(int i = 0; i < kerSize.height; i++){
		
		if(i+kerOffsetY<0) continue;
		if((i+kerOffsetY)>(WinSize.height-kerSize.height)) break;

		for(int j = 0; j < kerSize.width; j++){
		
			if(j+kerOffsetX<0) continue;
			if((j+kerOffsetX)>(WinSize.width-kerSize.width)) break;

			sum += fDevDelta[(i+kerOffsetY)*dstWidth+j+kerOffsetX+nNumber*(WinSize.width-kerSize.width+1)]*
				fDevKernel[Idx_kerBase+i*kerSize.width+j];
		}
	}

	fDevDeltaLower[Idx_s] = sum;
}

// Rotate the kernel in order to calculate the delta values of convolutio layers
inline __global__ void RotateKernel(
	float*				fKernel,
	NppiSize			kerSize,
	int					nCount						// the total number of kernels in the memory
){

	int Idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(Idx >= nCount) return;
	float fTemp = 0.0f;

	int Idx_base = Idx*kerSize.width*kerSize.height;

	for(int i = 0; i <= (kerSize.height-1)/2 ; i++){
		for(int j = 0; j < kerSize.width; j++){
		
			if((i==kerSize.height/2)&&(kerSize.width - j - 1 <= j)) return;

			fTemp = fKernel[Idx_base + i*kerSize.width+j];
			fKernel[Idx_base + i*kerSize.width+j] = fKernel[Idx_base + (kerSize.height-i-1)*kerSize.width+kerSize.width-j-1];
			fKernel[Idx_base + (kerSize.height-i-1)*kerSize.width+kerSize.width-j-1] = fTemp;
		}
	}
}

// The back propagation kernel function of multi-layer perceptron
// Being used in fully connected layers and output layer.
inline __global__ void PropDeltaMLP(
	cPixel*			fDevSrc,
	float*			fDevDeltaLower,
	float*			fDevDelta,
	float*			fDevWeights,		// The size of weights is nCount * iSize
	int				nCount,				// The count of higher layer delta
	NppiSize		iSize				// The count of lower layer delta
){
	int Idx = blockDim.x * blockIdx.x + threadIdx.x;

	if(Idx >= iSize.width * iSize.height) return;
	if(fDevSrc[Idx] == 0){
	
		fDevDeltaLower[Idx] = 0.0;
		return;
	}

	float sum = 0.0;

	for(int i = 0; i < nCount; i++){
	
		sum += fDevWeights[i * nCount + Idx] * fDevDelta[i];
	}

	fDevDeltaLower[Idx] = sum;
}

// The function back propogating the mask of max pooling 
// layers when perform learning of back propogation.
inline __global__ void PropMaxpooling(
	float*			fDevDeltaLower,
	float*			fDevDelta,
	bool*			bDevWeights,
	NppiSize		kerSize,
	NppiSize		MapSize
){
	int Idx = blockDim.x * blockIdx.x + threadIdx.x;

	if(Idx >= MapSize.width * MapSize.height) return;

	if(bDevWeights[Idx] == false) fDevDeltaLower[Idx] = 0.0;
	else {
	
		int Idx_sx = Idx % MapSize.width;
		int Idx_sy = Idx / MapSize.width;

		int Idx_dx = Idx_sx / kerSize.width;
		int Idx_dy = Idx_sy / kerSize.height;

		int Idx_d = Idx_dy * (MapSize.width / kerSize.width) + Idx_dx;

		fDevDeltaLower[Idx] = fDevDelta[Idx_d];
	} 
}

// Update the weights by input & delta with learning rate
// This kernel is using the formula of PRML
inline __global__ void UpdateDevWeights(
	cPixel*				DevSrc,
	float*				DevDelta,
	float*				fWeights,
	float				fLearnRate, 
	float				fLearnMomentum,
	float				fWeightDecay,
	int					nCount,			// size of input before extended
	int					iSize			// size of extended input and weights
){
	int Idx = blockDim.x * blockIdx.x + threadIdx.x;

	if(Idx >= iSize) return;

	int Idx_s = Idx % nCount;
	int Idx_d = Idx / nCount;

	if(DevSrc[Idx_s] == 0) return;

	// This is the update rule, using En = sigma(k){ ynk^tnk }
	fWeights[Idx] -= fWeightDecay*fLearnRate*fWeights[Idx] + fLearnMomentum*fLearnRate*((DevSrc[Idx_s] / 255.0f) * DevDelta[Idx_d]);
}

// Calculate the delta value between output and correct label
// We define the
inline __global__ void CalcOutputDelta(
	cPixel*		DevSrc,
	cPixel*		DevDst,
	float*		fDevDelta,
	int			nCount
){
	int Idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(Idx >= nCount) return;

	if(DevDst[Idx] > DevSrc[Idx]){
	
		fDevDelta[Idx] = (DevDst[Idx] - DevSrc[Idx]) / 255.0f;

	}else{

		fDevDelta[Idx] = (0.0f - (DevSrc[Idx] - DevDst[Idx])) / 255.0f;
	}
}

inline __global__ void softmax(
	float*				DevSrc,
	cPixel*				DevDst,
	int					nCount
){	
	int Idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(Idx >= nCount) return;

	double sum = 0.0;

	for(int i = 0; i < nCount; i++){
	
		sum += DevSrc[i];
	}

	DevDst[Idx] = DevSrc[Idx] / sum * 255; 
}

inline __global__ void AddUpFloat(
	float*		DevSrc,
	float*		DevDst,
	int			nCount,
	int			oSize
){
	int Idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(Idx >= oSize) return;
	float tmp = DevSrc[Idx * nCount];

	for(int i = 1; i < nCount; i++){
	
		tmp += DevSrc[Idx * nCount + i];
	}

	DevDst[Idx] = tmp;
}

// 1-D duplication of the first fully connected layer
// Tile the input and into nCount copies
inline __global__ void ExtInputKernel1D(
	cPixel*			DevSrc,
	cPixel*			DevDst,
	NppiSize		iSize,
	int				nCount
){
	int Idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(Idx >= iSize.width * iSize.height) return;
	int temp = DevSrc[Idx];

	for(int i = 0; i < nCount; i++){
	
		DevDst[Idx+i*iSize.width*iSize.height] = temp;
	}
}

// the original size of DevSrc is iSize
// This is a 2D method to copy the input into several copies
inline __global__ void ExtInputKernel(
	cPixel*			DevSrc,
	cPixel*			DevDst,
	NppiSize		iSize,
	NppiSize		oSize
){
	int Idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(Idx >= oSize.height * oSize.width) return;

	int Idx_ox = Idx % oSize.width;
	int Idx_oy = Idx / oSize.width;

	int Idx_ix = Idx_ox % iSize.width;
	DevDst[Idx] = DevSrc[Idx_oy * iSize.width + Idx_ix];
}

inline __global__ void Pointer(thrust::device_ptr<cPixel*> *DevInPtr){

	DevInPtr[threadIdx.x] += 0;
}

// Convert the temporary float to final image
inline __global__ void Convert2Img(
	float* DevSrc,
	cPixel* DevDst,
	int nCount
){
	int Idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(Idx >= nCount) return;

	if(DevSrc[Idx] > 255) DevDst[Idx] = 255;
	else DevDst[Idx] = DevSrc[Idx];
}

// Calculating the exp of the input
inline __global__ void Expo1D(
	float* DevSrc,
	float* DevDst,
	int nCount
){
	int Idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(Idx >= nCount) return;

	DevDst[Idx] = expf(DevSrc[Idx]);
}

inline __global__ void AddUpLines(
	float* DevSrc,
	int LineStep
){
	int Idx = threadIdx.x * LineStep;
	for(int i = 1; i < LineStep; i++){
	
		DevSrc[Idx] += DevSrc[Idx + i];
	}
}

// Adding up all DecSrc into one unit
inline __global__ void AddUp1D(
	float* DevSrc,
	cPixel* DevDst,
	int nCount,					// Adding up window size
	int nSize					// Size of the output image
){
	// Basic index
	int Idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(Idx >= nSize) return;

	// First element of this thread is drawn from source
	float res = DevSrc[Idx * nCount];

	for(int i = 1; i < nCount; i++){
	
		res += DevSrc[Idx * nCount + i];
	}

	//res += fDevOffsets[Idx];
	if(res >= 255.0f) DevDst[Idx] = 255;
	else DevDst[Idx] = res;
}

// Multiply the 1D weights
inline __global__ void MultiplyWeights1D(
	cPixel* DevSrc,
	float* DevDst,
	float* fWeights,
	int nCount					// the count of the pixels in the whole input
){
	int Idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(Idx > nCount) return;
	DevDst[Idx] =  DevSrc[Idx] * fWeights[Idx];
}

// The kernel for first fully-connected layer
// Transform dense image input into one level neuron
// parameter Idx_0 stands for the number of current image of all inputs
inline __global__ void CombineInput(
	cPixel* DevSrc,
	cPixel* DevDst,
	int Idx_0,
	NppiSize Size
){
	int Idx = Idx_0 * Size.width * Size.height ;
	DevDst[Idx + (threadIdx.y * Size.width) + threadIdx.x] = DevSrc[(threadIdx.y * Size.width) + threadIdx.x];
}

// Kernel for normal multilayer perceptron algorithm
// One thread for one pixel in the result layer
// 160 threads grouped in one block
inline __global__ void MLPKernel(
	cPixel* DevSrc,
	cPixel* DevDst,
	int SizeIn,
	float* fDevOffset,
	float* fKernel
){
	int Idx = blockIdx.x * blockDim.x + threadIdx.x;
	int tmp = 0;
	for(int i = 0; i < SizeIn; i++){
	
		tmp += DevSrc[i] * fKernel[i];
	}
	tmp += *fDevOffset;

	if(tmp > 255) tmp = 255;
	DevDst[Idx] = tmp;
}

inline __global__ void AddOffsets(
	cPixel*			DevSrc,
	float*			fDevOffsets,
	NppiSize		MapSize,
	NppiSize		WinSize
){
	int Idx = blockIdx.x * blockDim.x + threadIdx.x;

	// Count which window this thread is belonging to
	int Idx_wx = (Idx%MapSize.width)/WinSize.width;

	float tmp = DevSrc[Idx] + fDevOffsets[Idx_wx];
	if(tmp > 255.0f) DevSrc[Idx] = 255;
	else DevSrc[Idx] = tmp;
}

// The kernel function of RELU active function
inline __global__ void ReLU(
	cPixel* DevSrc,
	NppiSize MapSize
){
	int Idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(Idx >= MapSize.height * MapSize.width) return;

	if(DevSrc[Idx] < 0) DevSrc[Idx] = 0;
}

inline __global__ void maxpooling(
	cPixel*			DevSrc,
	cPixel*			DevDst,
	bool*			bDevWeights,
	NppiSize		SrcSize,
	NppiSize		DstSize,
	NppiSize		kerSize
){
	// Idx_dx : the base x position of this thread in the DevDst
	int Idx_d = blockIdx.x * blockDim.x + threadIdx.x;

	int Idx_dx = Idx_d % DstSize.width;
	// Idx_dy : the base y position of this thread in the DevDst
	int Idx_dy = Idx_d / DstSize.width;
	if(Idx_dy >= DstSize.height) return;

	int max = 0;
	int Idx_s = 0;
	int Idx_max = -1;

	for(int i = 0; i < kerSize.height; i++){

		if( Idx_dy * kerSize.height + i >= SrcSize.height) break;
		for(int j = 0; j < kerSize.width; j++){
		
			if( Idx_dx * kerSize.width + j >= SrcSize.width) break;
			// Idx_s : the one dimension position of this thread in the DevSrc
			Idx_s = (Idx_dy * kerSize.height + i) * SrcSize.width + (Idx_dx * kerSize.width + j);
			if( DevSrc[Idx_s] > max) {
				max = DevSrc[Idx_s];
				Idx_max = Idx_s;
			}
			bDevWeights[Idx_s] = false;
		}
	}
	bDevWeights[Idx_max] = true;
	DevDst[Idx_d] = max;
}

// The kernel function running by CUDA of maxpooling
/*inline __global__ void maxpooling(
	cPixel* DevSrc,
	cPixel* DevDst,
	NppiSize SrcSize,
	NppiSize DstSize,
	NppiSize kerSize
){

	int max = 0;
	int Idx_1 = 0;

	for(int i = 0; i < kerSize.width; i++){

		if(threadIdx.x * kerSize.width + i >= SrcSize.width) break;
		for(int j = 0; j < kerSize.height; j++){
		
			if(threadIdx.y * kerSize.height + j >= SrcSize.height) break;

			Idx_1 = (threadIdx.y * kerSize.height + j) * SrcSize.width + (threadIdx.x * kerSize.width + i);
			if(max < DevSrc[Idx_1]) max = DevSrc[Idx_1];
		}
	}
	DevDst[threadIdx.y * DstSize.width + threadIdx.x] = max;
}*/

//-----------------------------------------------------------------------------------------
//--------------------------------Host functions below-----------------------------------
//-----------------------------------------------------------------------------------------

void cFilterGroup::Initialize(short Level){

	cImages* TempOut;
	cImagesGPU* TempDevOut;

	sLevel = Level;
	sFilterMultiplier = GlobalValues.FilterMultipliers[sLevel];
	cFilter* TempFilter;
	Output = thrust::host_vector<cImages*>();
	DevOutput = thrust::host_vector<cImagesGPU*>();

	cudaEventCreate(&EventH2D);
	cudaEventCreate(&EventCalc);
	cudaEventCreate(&EventD2H);
	cudaEventCreate(&EventD2D);

	fDevOffsets = NULL;
	fOffsets = NULL;
	fDevTemp = NULL;
	fDevDelta = NULL;
	fDeltaTemp = NULL;
	DevInTemp = NULL;

	if(sLevel == 0) sInputCount = 1;
	else sInputCount = GlobalValues.FilterCounts[sLevel - 1];

	OutSize = GlobalValues.MapSizes[sLevel+1];

	DevInput = thrust::host_vector<cImagesGPU*>();

	if(Type == cGlobalValues::CONVOLUTION){
		
		nFilterCount = GlobalValues.FilterCounts[sLevel];
		MapSize = GlobalValues.MapSizes[sLevel];
		kerSize = GlobalValues.kerSizes[sLevel];

		OutSize.width *= nFilterCount;

		InitializeOffsets();

		TempOut = new cImages(OutSize.width, OutSize.height);
		Output.push_back(TempOut);

		TempDevOut = new cImagesGPU(*TempOut, true);
		DevOutput.push_back(TempDevOut);

		checkCudaErrors(cudaMalloc((float**)&fDevDelta, OutSize.height * OutSize.width * sizeof(float)));
		checkCudaErrors(cudaMemset(fDevDelta, 0, OutSize.height * OutSize.width * sizeof(float)));
		
		checkCudaErrors(cudaMalloc((float**)&fDevTemp, kerSize.width * nFilterCount * sizeof(float) * kerSize.height));
		checkCudaErrors(cudaMemset(fDevTemp, 0, kerSize.width * nFilterCount * sizeof(float) * kerSize.height));

		checkCudaErrors(cudaMalloc((float**)&fDevDeltaTemp, kerSize.width * nFilterCount * sizeof(float) * kerSize.height));
		checkCudaErrors(cudaMemset(fDevDeltaTemp, 0, kerSize.width * nFilterCount * sizeof(float) * kerSize.height));

		checkCudaErrors(cudaMalloc((float**)&fDeltaTemp, MapSize.width * MapSize.height * nFilterCount * sizeof(float)));
		checkCudaErrors(cudaMemset(fDeltaTemp, 0, MapSize.width * MapSize.height * nFilterCount * sizeof(float)));

		DevInTemp = new cImagesGPU(MapSize.width * nFilterCount, MapSize.height, true);

	}else if(Type == cGlobalValues::MAXPOOLING){
		
		nFilterCount = 1;
		MapSize = GlobalValues.MapSizes[sLevel];
		kerSize = GlobalValues.kerSizes[sLevel];
		OutSize.width *= sInputCount;
		MapSize.width *= sInputCount;

		TempOut = new cImages(OutSize.width, OutSize.height);
		Output.push_back(TempOut);

		TempDevOut = new cImagesGPU(*TempOut, true);
		DevOutput.push_back(TempDevOut);

		checkCudaErrors(cudaMalloc((float**)&fDevDelta, OutSize.width * OutSize.height * sizeof(float)));
		checkCudaErrors(cudaMemset(fDevDelta, 0, OutSize.height * OutSize.width * sizeof(float)));

	}else if(Type == cGlobalValues::FULLYCONNECTED){

		nFilterCount = 1;
	
		if(sLevel == GlobalValues.sConvLayerCount + GlobalValues.sPoolLayerCount){
			
			MapSize.width = (GlobalValues.MapSizes[sLevel].height) * (GlobalValues.MapSizes[sLevel].width) *
				(GlobalValues.FilterCounts[sLevel-1]) * (GlobalValues.sFullConnNeuronCount);
			MapSize.height = 1;

		}else{
		
			MapSize.width = GlobalValues.sFullConnNeuronCount * GlobalValues.sFullConnNeuronCount;
			MapSize.height = 1;

		}
		
		DevInTemp = new cImagesGPU( MapSize.width, MapSize.height, true);

		checkCudaErrors(cudaMalloc((float**)&fDevTemp, MapSize.height * MapSize.width * sizeof(float)));
		checkCudaErrors(cudaMalloc((float**)&fDevDelta, GlobalValues.sFullConnNeuronCount * sizeof(float)));
		checkCudaErrors(cudaMemset(fDevDelta, 0, GlobalValues.sFullConnNeuronCount * sizeof(float)));

		checkCudaErrors(cudaMalloc((float**)&fDeltaTemp, MapSize.width / GlobalValues.sFullConnNeuronCount));

		OutSize.width = GlobalValues.sFullConnNeuronCount;
		OutSize.height = 1;

		//InitializeOffsets();
		// Output setting of fully connected layers
		TempOut = new cImages(OutSize.width, OutSize.height);
		Output.push_back(TempOut);

		TempDevOut = new cImagesGPU(*TempOut, true);
		DevOutput.push_back(TempDevOut);

	}else{

		nFilterCount = 1;
	
		MapSize.width = GlobalValues.sFullConnNeuronCount * GlobalValues.sOutputCount; 
		MapSize.height = 1;

		TempOut = new cImages(GlobalValues.sOutputCount, 1);
		Output.push_back(TempOut);

		TempDevOut = new cImagesGPU(*TempOut, true);
		DevOutput.push_back(TempDevOut);

		OutSize.width = GlobalValues.sOutputCount;
		OutSize.height = 1;

		DevInTemp = new cImagesGPU( MapSize.width, MapSize.height, true);

		checkCudaErrors(cudaMalloc((float**)&fDevTemp, MapSize.height * MapSize.width * sizeof(float)));
		checkCudaErrors(cudaMalloc((float**)&fDevDelta, GlobalValues.sOutputCount * sizeof(float)));
		checkCudaErrors(cudaMemset(fDevDelta, 0, GlobalValues.sOutputCount * sizeof(float)));
	}

	for(int i = 0; i < nFilterCount; i++){
	
		TempFilter = new cFilter();
		
		TempFilter->kerSize = kerSize;
		TempFilter->MapSize = MapSize;
		TempFilter->sInputCount = sInputCount;

		if(Type == cGlobalValues::CONVOLUTION){
	
			Anchor.x = kerSize.width;
			Anchor.y = kerSize.height;

			TempFilter->Type = cGlobalValues::CONVOLUTION;

		}else if(Type == cGlobalValues::MAXPOOLING){

			TempFilter->Type = cGlobalValues::MAXPOOLING;

		}else if(Type ==cGlobalValues::FULLYCONNECTED){
			
			TempFilter->Type = cGlobalValues::FULLYCONNECTED;

		}else{
			
			TempFilter->Type = cGlobalValues::OUTPUT;
		}
		
		TempFilter->Initialize(sLevel);

		Filters.push_back(TempFilter);
	}
	if(Type == cGlobalValues::CONVOLUTION) CopyWeights();
}

void cFilterGroup::InitializeOffsets(){

	if(Type == cGlobalValues::CONVOLUTION){
	
		checkCudaErrors(cudaMallocHost((void**)&fOffsets, nFilterCount * sizeof(float)));

		for(int i = 0; i < nFilterCount; i++){
		
			fOffsets[i] = fRandom(GlobalValues.fMinRandWeight, GlobalValues.fMaxRandWeight);
		}

		if(sLevel == 0){
		
			for(int i = 0; i < nFilterCount; i++){
		
				fOffsets[i] = fRandom(15.f, 35.f);
			}
		}

		checkCudaErrors(cudaMalloc((float**)&fDevOffsets, nFilterCount * sizeof(float)));
		checkCudaErrors(cudaMemcpy(fDevOffsets, fOffsets, nFilterCount * sizeof(float), cudaMemcpyHostToDevice));

	}else if(Type == cGlobalValues::FULLYCONNECTED){
	
		checkCudaErrors(cudaMallocHost((void**)&fOffsets, GlobalValues.sFullConnNeuronCount * sizeof(float)));

		for(int i = 0; i < GlobalValues.sFullConnNeuronCount; i++){
		
			fOffsets[i] = 0;//fRandom(GlobalValues.fMinRandWeight, GlobalValues.fMaxRandWeight);
		}

		checkCudaErrors(cudaMalloc((float**)&fDevOffsets, GlobalValues.sFullConnNeuronCount * sizeof(float)));
		checkCudaErrors(cudaMemcpy(fDevOffsets, fOffsets, GlobalValues.sFullConnNeuronCount * sizeof(float), cudaMemcpyHostToDevice));
	}
}

void cFilterGroup::Compute(short sNumber){

	//Npp32f* fDevKernel = NULL;
	//Npp32s* nDevKernel = NULL;
	//cImagesGPU* DevDst = NULL;
	//cImagesGPU* DevSrc = NULL;
	//cImages* TempOut = NULL;
	Npp32s nDivisor = 0;
	NppiSize ROISize, oROISize;
	NppStatus nStatus;
	dim3 threadsPerBlock = 0;
	dim3 blockCount = 0;

	if(Type == cGlobalValues::FULLYCONNECTED){

		threadsPerBlock = 160;
		blockCount = ceil((double) MapSize.width * MapSize.height * 1.0 / threadsPerBlock.x);
		
		MultiplyWeights1D<<<blockCount, threadsPerBlock, 0, *CalcStream>>>( 
			DevInput[0]->data(), fDevTemp, Filters[0]->fDevWeights, MapSize.height * MapSize.width);

		blockCount = ceil((float) GlobalValues.sFullConnNeuronCount * 1.0 / threadsPerBlock.x);
		
		int InputSize = MapSize.width / GlobalValues.sFullConnNeuronCount;

		AddUp1D<<<blockCount, threadsPerBlock, 0, *CalcStream>>>( 
			fDevTemp, DevOutput[0]->data(), InputSize, GlobalValues.sFullConnNeuronCount);

		checkCudaErrors(cudaEventRecord( EventCalc, *CalcStream));

	}else if(Type == cGlobalValues::MAXPOOLING){
	
		threadsPerBlock = 160;

		ROISize.height = DevInput[0]->height();
		ROISize.width = DevInput[0]->width();

		OutSize.height = ROISize.height/kerSize.height;
		OutSize.width = ROISize.width/kerSize.width;

		blockCount = ceil(OutSize.height * OutSize.width * 1.0 / threadsPerBlock.x);

		maxpooling<<<blockCount, threadsPerBlock, 0, *CalcStream>>>( 
			DevInput[0]->data(), DevOutput[0]->data(), Filters[0]->bDevWeights, ROISize, OutSize, kerSize);

		checkCudaErrors(cudaEventRecord( EventCalc, *CalcStream));

	}else if(Type == cGlobalValues::CONVOLUTION){

		OutSize.width =DevOutput[0]->width();
		OutSize.height = DevOutput[0]->height();

		ROISize.height = MapSize.height;
		ROISize.width = MapSize.width;
		
		Anchor.x = kerSize.width-1;
		Anchor.y = kerSize.height-1;

		oROISize.height = MapSize.height - kerSize.height + 1;
		oROISize.width = MapSize.width - kerSize.width + 1;

		int threadsPerBlock =  192;
		int blockCount = ceil((double) OutSize.width * OutSize.height * 1.0 / threadsPerBlock);

		ConvoKernel<<<blockCount, threadsPerBlock, 0, *CalcStream>>>
			( DevInput[0]->data(), DevOutput[0]->data(), fDevTemp, oROISize, kerSize, Filters.size());
		
		// Main loop for convolution of each filter
		/*for(int i = 0; i < Filters.size(); i++){

			if(GlobalValues.bFloatWeight){

				// 8 bit single channel image
				nppiFilter32f_8u_C1R( DevInput[0]->data(i * ROISize.width), DevInput[0]->pitch(), DevOutput[0]->data(i * oROISize.width), 
					DevOutput[0]->pitch(), ROISize, Filters[i]->fDevWeights, kerSize, Anchor);
			}
			
			//Trace( fDevTemp, fSize);
			// Copy the result of convolution back to the outputs of groups
		}	*/

		//threadsPerBlock = 160;
		//blockCount = DevOutput[0]->width() * DevOutput[0]->height() / threadsPerBlock.x; // According to test plus 5 gives the right result. It's a myth
		
		//AddOffsets<<<blockCount, threadsPerBlock, 0, *CalcStream>>>( DevOutput[0]->data(), fDevOffsets, OutSize, oROISize); 
		
		checkCudaErrors(cudaEventRecord(EventCalc, *CalcStream));

	}else if(Type == cGlobalValues::OUTPUT){

		threadsPerBlock = 160;
		blockCount = ceil((double) MapSize.width * MapSize.height * 1.0 / threadsPerBlock.x);
		int nCount = MapSize.width * MapSize.height;
		float* fOutTmp;

		checkCudaErrors(cudaMalloc((float**)&fOutTmp, sizeof(float) * GlobalValues.sOutputCount));
		
		MultiplyWeights1D<<<blockCount, threadsPerBlock, 0, *CalcStream>>>( 
			DevInput[0]->data(), fDevTemp, Filters[0]->fDevWeights, MapSize.width * MapSize.height);

		blockCount = ceil((double) GlobalValues.sOutputCount * 1.0 / threadsPerBlock.x);

		/*AddUp1D<<<blockCount, threadsPerBlock, 0, *CalcStream>>>( 
			fDevTemp, DevOutput[0]->data(), GlobalValues.sFullConnNeuronCount, GlobalValues.sOutputCount);*/

		AddUpFloat<<< blockCount, threadsPerBlock, 0, *CalcStream>>>( 
			fDevTemp, fOutTmp, GlobalValues.sFullConnNeuronCount, GlobalValues.sOutputCount);

		Expo1D<<< blockCount, threadsPerBlock, 0, *CalcStream>>>( 
			fOutTmp, fOutTmp, GlobalValues.sOutputCount);

		softmax<<< blockCount, threadsPerBlock, 0, *CalcStream>>>( 
			fOutTmp, DevOutput[0]->data(), GlobalValues.sOutputCount);

		checkCudaErrors(cudaEventRecord( EventCalc, *CalcStream));
		checkCudaErrors(cudaEventSynchronize( EventCalc));

		cudaFree(fOutTmp);
	}
}

// Alloc the big all-in-one output
void cFilterGroup::DevOutAlloc(){

	NppiSize outSize;
	outSize.height = MapSize.height / kerSize.height;
	outSize.width = MapSize.width / kerSize.width;

	if(DevOut != NULL) cudaFree(DevOut);

	checkCudaErrors(cudaMalloc((float**)&DevOut, outSize.width * outSize.height * sInputCount * sizeof(float)));
}

// For output layer, setting target label to the layer
void cFilterGroup::SetTarget(cImagesGPU* Target){

	int threadsPerBlock = 160;
	int blockCount = ceil((double) GlobalValues.sOutputCount * 1.0 / threadsPerBlock);

	CalcOutputDelta<<<blockCount, threadsPerBlock, 0, *CalcStream>>>( 
		Target->data(), DevOutput[0]->data(), fDevDelta, GlobalValues.sOutputCount);
	
	checkCudaErrors(cudaEventRecord( EventCalc, *CalcStream));

	checkCudaErrors(cudaEventSynchronize( EventCalc));
}

// Only allocate separated dev memory
void cFilterGroup::DevOutputAlloc(){

	if(!DevOutput.empty()){
	
		for(int i = 0; i < DevOutput.size(); i++){
		
			if(DevOutput[i] != NULL){
			
				delete DevOutput[i];
				DevOutput[i] = NULL;
			}
		}
		DevOutput.clear();
	}

	cImagesGPU* DevTmp;
	for(int i = 0; i < Output.size(); i++){
	
		DevTmp = new cImagesGPU( Output[i]->width(), Output[i]->height());
		DevOutput.push_back( DevTmp);
	}
}

// Clear the input & output of device and re-allocate the device memory of this group
void cFilterGroup::DeviceAlloc(){

	cImagesGPU* DevSrc;

	for(int i = 0; i < Input.size(); i++){
	
		DevSrc = new cImagesGPU(*Input[i]);
		DevInput.push_back(DevSrc);
	}
	if(!(DevOutput.empty())){
	
		for(int i = 0; i < DevOutput.size(); i++){
		
			if(DevOutput[i] != NULL){
			
				delete DevOutput[i];
				DevOutput[i] = NULL;
			}
		}
		DevOutput.clear();
	}

	for(int i = 0; i < Output.size(); i++){
	
		DevSrc = new cImagesGPU(Output[i]->width(), Output[i]->height());
		DevOutput.push_back(DevSrc);
	}
}

void cFilterGroup::ClearGPUOutput(){

	for(int i = 0; i < DevOutput.size(); i++){
	
		if(DevOutput[i] != NULL){
		
			delete DevOutput[i];
			DevOutput[i] = NULL;
		}
	}

	DevOutput.clear();
}

/*void cFilterGroup::ClearGPUTmp(){

	for(int i = 0; i < DevTmp.size(); i++){
	
		if(DevTmp[i] != NULL){
		
			delete DevTmp[i];
			DevTmp[i] = NULL;
		}
	}

	DevTmp.clear();
}*/

void cFilterGroup::SetInputPointer(){

	cPixel** tmp;

	checkCudaErrors(cudaMalloc((cPixel***)&tmp, DevInput.size() * sizeof(cPixel*)));

	for(int i = 0; i < DevInput.size(); i++){
	
		tmp[i] = DevInput[i]->data();
	}

	DevInPtr = thrust::device_pointer_cast(tmp);
}

void cFilterGroup::SetOutputPointer(){

	cPixel** tmp;

	checkCudaErrors(cudaMalloc((cPixel***)&tmp, DevOutput.size() * sizeof(cPixel*)));

	for(int i = 0; i < DevOutput.size(); i++){
	
		tmp[i] = DevOutput[i]->data();
	}

	DevOutPtr = thrust::device_pointer_cast(tmp);
}

void cFilterGroup::ClearGPUInput(){

	for(int i = 0; i < DevInput.size(); i++){
	
		if(DevInput[i] != NULL){
		
			delete DevInput[i];
			DevInput[i] = NULL;
		}
	}

	DevInput.clear();
}

void cFilterGroup::ExtendInput(){

	int threadsPerBlock;
	int blockCount;

	NppiSize oSize;
	NppiSize iSize;

	//oSize.width = iSize.width * DevInput.size(); Always one input

	if(Type == cGlobalValues::CONVOLUTION){
	
		iSize.height = InSize.height;
		iSize.width = InSize.width;

		oSize = iSize;

		oSize.width *= sFilterMultiplier;

	}else{

		iSize.height = DevInput[0]->height();
		iSize.width = DevInput[0]->width();

		oSize = iSize;

		if(Type == cGlobalValues::FULLYCONNECTED) oSize.width = GlobalValues.sFullConnNeuronCount*GlobalValues.sFullConnNeuronCount;
		else oSize.width = GlobalValues.sFullConnNeuronCount * GlobalValues.sOutputCount;
	}

	threadsPerBlock = 160;
	blockCount = ceil((double) oSize.height * oSize.width * 1.0 / threadsPerBlock);

	for(int i = 0; i < sFilterMultiplier; i++){

		checkCudaErrors(cudaMemcpy2DAsync( DevInTemp->data(i * iSize.width), DevInTemp->pitch(),
			DevInput[0]->data(), DevInput[0]->pitch(), iSize.width * sizeof(cPixel) , iSize.height * sizeof(cPixel),
			cudaMemcpyDeviceToDevice, *D2DStream)); 
	}
	
	if(Type != cGlobalValues::CONVOLUTION){
	
		checkCudaErrors(cudaEventRecord(EventD2D, *D2DStream));
		checkCudaErrors(cudaStreamWaitEvent( *CalcStream, EventD2D, 0));
	}
	else{
		checkCudaErrors(cudaEventRecord(EventD2D, *D2DStream));
		checkCudaErrors(cudaStreamWaitEvent( *CalcStream, EventD2D,0));
	}
	
	DevInput = thrust::host_vector<cImagesGPU*>();
	DevInput.push_back(DevInTemp);
}

void cFilterGroup::ExtendInput1D(){

	int threadsPerBlock;
	int blockCount;

	//oSize.width = iSize.width * DevInput.size(); Always one input

	threadsPerBlock = 160;
	blockCount = ceil((double) InSize.height * InSize.width * 1.0 / threadsPerBlock);

	ExtInputKernel1D<<< blockCount, threadsPerBlock, 0, *D2DStream>>>( 
		DevInput[0]->data(), DevInTemp->data(), InSize, GlobalValues.sFullConnNeuronCount);

	checkCudaErrors(cudaEventRecord( EventD2D, *D2DStream));
	checkCudaErrors(cudaStreamWaitEvent( *CalcStream, EventD2D, 0));
	
	DevInput = thrust::host_vector<cImagesGPU*>();
	DevInput.push_back(DevInTemp);
}

void cFilterGroup::SetDeviceInput(cImagesGPU* devInputImage){


}

// Asychronizely copy inputs from CPU to GPU
// Need to synchronize explictly
void cFilterGroup::CopyResults(){

	npp::Image::Size SrcSize;

	SrcSize.nHeight = DevOutput[0]->height();
	SrcSize.nWidth = DevOutput[0]->width();

	checkCudaErrors( cudaStreamWaitEvent( *D2HStream, EventCalc, 0));

	for(int i = 0; i < DevOutput.size(); i++){

		checkCudaErrors(cudaMemcpyAsync(Output[i]->data(), DevOutput[i]->data(), 
			SrcSize.nHeight * SrcSize.nWidth * sizeof(cPixel),
			cudaMemcpyDeviceToHost, *D2HStream));
	}

	checkCudaErrors( cudaEventRecord( EventD2H, *D2HStream));
}

// Judging the situation of the filter groups and 
// decide to copy inputs combined or not
void cFilterGroup::CopyInputs(){

	cImagesGPU* DevTemp;
	npp::Image::Size SrcSize;
	
	SrcSize.nHeight = Input[0]->height();
	SrcSize.nWidth = Input[0]->width();

	if(DevInput.empty()) {
		DevInput.push_back( new cImagesGPU(*Input[0]));
		return;
	}

	
	// No event record here! Need explictly record in the layer
}

void cFilterGroup::Clean(){


}

void cFilterGroup::CopyWeightsH2D(){

	for(int i = 0; i < Filters.size(); i++){
	
		Filters[i]->CopyWeightsH2D();
	}
	checkCudaErrors(cudaEventRecord( EventH2D, *H2DStream));
}

void cFilterGroup::CopyWeightsD2H(){

	for(int i = 0; i < Filters.size(); i++ ){
	
		Filters[i]->CopyWeightsD2H();
	}
	//checkCudaErrors(cudaEventRecord( EventD2H, *D2HStream));
}

void cFilterGroup::TraceDevInput(){

	std::string filePath = std::string();
	char* cpath;
	cpath = (char*)malloc(20 * sizeof(char));

	cImages* HostDst = NULL;
	npp::Image::Size size;
																																			
	for(int i = 0; i < DevInput.size(); i++){

		size.nHeight = DevInput[i]->height();
		size.nWidth = DevInput[i]->width();

		HostDst = new cImages(size);

		DevInput[i]->copyTo( HostDst->data(), HostDst->pitch());

		itoa(sLevel, cpath, 10);
	
		filePath = INPUT_SAVE_PATH + std::string("_Level") + std::string(cpath);
		
		itoa(i, cpath, 10);

		filePath += "_No_" + std::string(cpath) + ".pgm";

		npp::saveImage( filePath, *HostDst);
	}

	free(cpath);
}

void cFilterGroup::TraceDevOutput(int col){

	std::string filePath = std::string();
	char* cpath;
	cpath = (char*)malloc(20 * sizeof(char));

	cImages* HostDst = NULL;
	npp::Image::Size size;
																																			
	for(int i = 0; i < DevOutput.size(); i++){

		size.nHeight = DevOutput[i]->height();
		size.nWidth = DevOutput[i]->width();

		HostDst = new cImages(size);

		DevOutput[i]->copyTo( HostDst->data(), HostDst->pitch());

		itoa(col, cpath, 10);

		filePath = OUTPUT_SAVE_PATH + std::string("_Column") + std::string(cpath);

		itoa(sLevel, cpath, 10);
	
		filePath += std::string("_Level") + std::string(cpath);
		
		itoa(i, cpath, 10);

		filePath += "_No_" + std::string(cpath) + ".pgm";

		npp::saveImage( filePath, *HostDst);
	}
	free(cpath);
}

// iSize is the number of deltas in the lower layer
void cFilterGroup::CalcDelta(float* fDevDeltaLower, NppiSize iSize){

	int threadsPerBlock = 160;
	int blockCount, nCount;

	if(Type == cGlobalValues::MAXPOOLING){
		
		blockCount = ceil((double) iSize.width * iSize.height * 1.0 / threadsPerBlock);
	
		PropMaxpooling<<<blockCount, threadsPerBlock, 0, *CalcStream>>>( 
			fDevDeltaLower, fDevDelta, Filters[0]->bDevWeights, kerSize, MapSize);

	}else if(Type == cGlobalValues::CONVOLUTION){

		NppiSize fSize, tempSize, kSize;

		fSize.width = MapSize.width * nFilterCount;
		fSize.height = MapSize.height;

		//CopyWeights();

		kSize.width = kerSize.width;
		kSize.height = kerSize.height * nFilterCount;
		
		//TraceWideWeights();

		//Trace(fDevTemp, kSize);

		blockCount = ceil((double)Filters.size() * 1.0 / threadsPerBlock);

		RotateKernel<<<blockCount, threadsPerBlock, 0, *CalcStream>>>( 
			fDevTemp, kerSize, Filters.size());
		
		//Trace(fDevTemp, kSize);

		//TraceWideWeights();

		blockCount = ceil((double)fSize.width * fSize.height * 1.0f / threadsPerBlock);

		PropDeltaConvo<<<blockCount, threadsPerBlock, 0, *CalcStream>>>
			( DevInput[0]->data(), fDeltaTemp, fDevDelta, fDevTemp, MapSize, kerSize, Filters.size());

		//Trace(fDeltaTemp, fSize);

		blockCount = ceil((double) iSize.width * iSize.height * 1.0 / threadsPerBlock);

		CombineDeltaConvo<<<blockCount, threadsPerBlock, 0, *CalcStream>>>( 
			DevInput[0]->data(), fDeltaTemp, fDevDeltaLower, iSize, fSize, sFilterMultiplier);

		blockCount = ceil((double)Filters.size() * 1.0 / threadsPerBlock);

		RotateKernel<<<blockCount, threadsPerBlock, 0, *CalcStream>>>( 
			fDevTemp, kerSize, Filters.size());

		//Trace(fDevTemp, kSize);

		//Trace(fDevDeltaLower, iSize);

	}else if(Type == cGlobalValues::FULLYCONNECTED){

		blockCount = ceil((double) iSize.width * iSize.height * 1.0 / threadsPerBlock);
	
		PropDeltaMLP<<<blockCount, threadsPerBlock, 0, *CalcStream>>>( 
			DevInput[0]->data(), fDevDeltaLower, fDevDelta, Filters[0]->fDevWeights, GlobalValues.sFullConnNeuronCount, iSize);
	}else{

		blockCount = ceil((double) iSize.width * iSize.height * 1.0 / threadsPerBlock);
	
		PropDeltaMLP<<<blockCount, threadsPerBlock, 0, *CalcStream>>>( 
			DevInput[0]->data(), fDevDeltaLower, fDevDelta, Filters[0]->fDevWeights, GlobalValues.sOutputCount, iSize);
	}
}

void cFilterGroup::TraceWideWeights(){

	float* fTemp = NULL;

	fTemp = (float*) malloc(kerSize.width * kerSize.height * nFilterCount * sizeof(float));

	checkCudaErrors( cudaMemcpy( fTemp, fDevTemp, kerSize.width * nFilterCount * sizeof(float) *
		kerSize.height, cudaMemcpyDeviceToHost));

	fprintf(logFile, "||| Starting tracing wide weights No.1 filter group:\n");

	for(int i = 0; i < kerSize.height * nFilterCount; i++){
		for(int j = 0; j < kerSize.width; j++){
			fprintf(logFile, "%f ", fTemp[i * (kerSize.width) + j]);
		}
		fprintf(logFile, "\n");
	}

	fprintf(logFile, "||| Wide weights tracing finished.\n\n");

	free(fTemp);
}

void cFilterGroup::CopyWeights(){
	
	for(int i = 0; i < Filters.size(); i++){
	
		checkCudaErrors(cudaMemcpyAsync(&fDevTemp[i*kerSize.width*kerSize.height], Filters[i]->fDevWeights, kerSize.width * kerSize.height * sizeof(float), cudaMemcpyDeviceToDevice, *D2DStream));
	}

	cudaEventRecord( EventD2D, *D2DStream);
	checkCudaErrors(cudaStreamWaitEvent( *CalcStream, EventD2D, 0));
}

void cFilterGroup::CopyDevInput(cImagesGPU* LowerDevOutput, npp::Image::Size iSize){

	for(int i = 0; i < sFilterMultiplier; i++){
	
		checkCudaErrors(cudaMemcpy2DAsync( DevInput[0]->data(i * iSize.nWidth), DevInput[0]->pitch(),
			LowerDevOutput->data(), LowerDevOutput->pitch(), iSize.nWidth * sizeof(cPixel),
			iSize.nHeight * sizeof(cPixel), cudaMemcpyDeviceToDevice, *D2DStream));
	} 
	
	if(Type != cGlobalValues::CONVOLUTION){
	
		checkCudaErrors(cudaEventRecord(EventD2D, *D2DStream));
		checkCudaErrors(cudaStreamWaitEvent( *CalcStream, EventD2D, 0));
	}
	else{
		checkCudaErrors(cudaEventRecord(EventD2D, *D2DStream));
		checkCudaErrors(cudaStreamWaitEvent( *CalcStream, EventD2D,0));
	}
}

void cFilterGroup::CopyDevInput1D(cImagesGPU* LowerDevOutput, npp::Image::Size iSize){

	int threadsPerBlock = 160;
	int blockCount = 0;

	NppiSize InSize;
	InSize.width = iSize.nWidth;
	InSize.height = iSize.nHeight;

	blockCount = ceil((double)InSize.width * InSize.height * 1.0 / threadsPerBlock);

	ExtInputKernel1D<<<blockCount, threadsPerBlock, 0, *D2DStream>>>
		( LowerDevOutput->data(), DevInput[0]->data(), InSize, GlobalValues.sFullConnNeuronCount);

	checkCudaErrors(cudaEventRecord( EventD2D, *D2DStream));
	checkCudaErrors(cudaStreamWaitEvent(*CalcStream, EventD2D, 0));
}

void cFilterGroup::UpdateConvoWeights(){

	if(fDevTemp == NULL) {

		StopErrorMsg("FilterGroup: SaveWeights error, null pointer.");

		return;
	}

	for(int i = 0; i < Filters.size(); i++){
	
		checkCudaErrors(cudaMemcpyAsync( Filters[i]->fDevWeights, &fDevTemp[i*kerSize.width*kerSize.height], kerSize.width * kerSize.height * sizeof(float), cudaMemcpyDeviceToDevice, *D2DStream));
	}
}

// For consistency with PRML, using delta to denote the partial differentiate of En to a
// En denotes the error function of input xn and a denotes the input part of activation function
void cFilterGroup::Train(int nCount){

	int threadsPerBlock = 160;

	NppiSize iSize;
	iSize.width = DevInput[0]->width();
	iSize.height = DevInput[0]->height();

	int blockCount = 0;
	int weightSize = Filters.size() * kerSize.width * kerSize.height;

	if(Type == cGlobalValues::CONVOLUTION){

		//if( sLevel == 0) CopyWeights(); There is no need to copy again

		blockCount = ceil((double)Filters.size() * 1.0 / threadsPerBlock);

		NppiSize oSize;
		oSize.width = MapSize.width - kerSize.width + 1;
		oSize.height = MapSize.height - kerSize.height + 1;

		blockCount = ceil((double) weightSize * 1.0 / threadsPerBlock);

		UpdateWeightsDeltaConvo<<<blockCount, threadsPerBlock, 0, *CalcStream>>>( 
			DevInput[0]->data(), fDeltaTemp, fDevDeltaTemp, kerSize, oSize, Filters.size());

		UpdateWeightsConvo<<<blockCount, threadsPerBlock, 0, *CalcStream>>>( fDevTemp, fDevDeltaTemp, GlobalValues.fWeightDecay, GlobalValues.fLearnStep,
			GlobalValues.fLearnMomentum, weightSize);

		//UpdateConvoWeights();

	}else if(Type == cGlobalValues::MAXPOOLING){
	
		return;
	}else if(Type == cGlobalValues::FULLYCONNECTED){

		blockCount = ceil((double) MapSize.width * MapSize.height * 1.0 / threadsPerBlock);
	
		UpdateDevWeights<<<blockCount, threadsPerBlock, 0, *CalcStream>>>( 
			DevInput[0]->data(), fDevDelta, Filters[0]->fDevWeights, GlobalValues.fLearnStep,
			GlobalValues.fLearnMomentum, GlobalValues.fWeightDecay, nCount, iSize.width * iSize.height);

	}else{

		blockCount = ceil((double) MapSize.width *  MapSize.height * 1.0 / threadsPerBlock);
	
		UpdateDevWeights<<<blockCount, threadsPerBlock, 0, *CalcStream>>>( 
			DevInput[0]->data(), fDevDelta, Filters[0]->fDevWeights, GlobalValues.fLearnStep,
			GlobalValues.fLearnMomentum, GlobalValues.fWeightDecay, nCount, iSize.width * iSize.height);
	}
}

// Saving weights to the hard disk
// Directly write into the weight file in binary way
void cFilterGroup::SaveWeights(){

	if(Type == cGlobalValues::CONVOLUTION){
	
		float* fTemp = NULL;
		size_t kSize = kerSize.width * kerSize.height * nFilterCount * sizeof(float);

		//CopyWeights();

		fTemp = (float*)malloc(kSize);

		checkCudaErrors(cudaMemcpy( fTemp, fDevTemp, kSize, cudaMemcpyDeviceToHost));

		size_t wCount = fwrite( fTemp, kSize, 1, weightsFile);
		
		free(fTemp);

	}else if(Type == cGlobalValues::MAXPOOLING){
	
		bool* bTemp = NULL;
		size_t kSize = MapSize.width * MapSize.height * sizeof(bool);

		bTemp = (bool*)malloc(kSize);

		checkCudaErrors(cudaMemcpy( bTemp, Filters[0]->bDevWeights, kSize, cudaMemcpyDeviceToHost));

		size_t wCount = fwrite( bTemp, kSize, 1, weightsFile);

		free(bTemp);

	}else{
	
		float* fTemp = NULL;
		size_t kSize = MapSize.width * MapSize.height * sizeof(float);

		fTemp = (float*)malloc( kSize);

		checkCudaErrors(cudaMemcpy( fTemp, Filters[0]->fDevWeights, kSize, cudaMemcpyDeviceToHost));

		size_t wCount = fwrite( fTemp, kSize, 1, weightsFile);
		
		free(fTemp);
	}
}

void cFilterGroup::LoadWeights(){

	if(Type == cGlobalValues::CONVOLUTION){
	
		float* fTemp = NULL;
		size_t kSize = kerSize.width * kerSize.height * nFilterCount * sizeof(float);

		fTemp = (float*)malloc(kSize);

		size_t rCount = fread( fTemp, kSize, 1, weightsFile);
		
		checkCudaErrors(cudaMemcpy( fDevTemp, fTemp, kSize, cudaMemcpyHostToDevice));

		UpdateConvoWeights();

		free(fTemp);

	}else if(Type == cGlobalValues::MAXPOOLING){
	
		bool* bTemp = NULL;
		size_t kSize = MapSize.width * MapSize.height * sizeof(bool);

		bTemp = (bool*)malloc(kSize);

		size_t rCount = fread( bTemp, kSize, 1, weightsFile);
		
		checkCudaErrors(cudaMemcpy( Filters[0]->bDevWeights, bTemp, kSize, cudaMemcpyHostToDevice));

		free(bTemp);

	}else{
	
		float* fTemp = NULL;
		size_t kSize = MapSize.width * MapSize.height * sizeof(float);

		fTemp = (float*)malloc(kSize);

		size_t rCount = fread( fTemp, kSize, 1, weightsFile);
		
		checkCudaErrors(cudaMemcpy( Filters[0]->fDevWeights, fTemp, kSize, cudaMemcpyHostToDevice));

		free(fTemp);
	}
}

void cFilterGroup::Trace(short sNumber){

	fprintf(logFile, "||| Starting tracing No.%d filter group:\n", sNumber);
	fprintf(logFile, "||| Filters count:%d.\n", Filters.size());

	if(!GlobalValues.bTraceFilter){
	
		fprintf(logFile, "||| Filter group tracing finished.\n|||\n");
		return;
	}
	for(int i = 0; i < Filters.size(); i++){
	
		Filters[i]->Trace(i+1);
	}

	fprintf(logFile, "||| Filter group tracing finished.\n|||\n");
}

void cFilterGroup::Trace(float* fDevTemp, NppiSize fSize){

	FILE* logTemp = NULL;

	float* fTemp = NULL;
	fTemp = (float*)malloc(fSize.width * fSize.height * sizeof(float));
	
	checkCudaErrors(cudaMemcpy( fTemp, fDevTemp, fSize.width * fSize.height * sizeof(float),
		cudaMemcpyDeviceToHost));

	logTemp = fopen( LOG_FILE_TEMP, "w");

	fprintf(logTemp, "||| Starting tracing float block:\n");
	fprintf(logTemp, "||| Block size - width: %d height: %d.\n", fSize.width, fSize.height);

	for(int i = 0; i < fSize.height; i++){
		for(int j = 0; j < fSize.width; j++){
		
			fprintf( logTemp, "%f ", fTemp[i * fSize.width + j]);
		}
		fprintf(logTemp, "\n");
	}

	fprintf(logTemp, "||| Float block tracing finished.\n|||\n");

	free(fTemp);
	fclose(logTemp);
}

void cFilterGroup::TraceDelta(FILE* log){

	float* fDelta;
	int iSize = OutSize.width * OutSize.height;

	fDelta = (float*)malloc(iSize * sizeof(float));

	checkCudaErrors(cudaMemcpy( fDelta, fDevDelta, iSize * sizeof(float), cudaMemcpyDeviceToHost));

	fprintf(log, "||| Starting tracing No.1 filter group:\n");
	fprintf(log, "||| Deltas count:%d.\n", OutSize.width * OutSize.height);

	for(int i = 0; i < OutSize.height; i++){
		for(int j = 0; j < OutSize.width; j++){
		
			fprintf( log, "%f ", fDelta[i * OutSize.width + j]);
		}
		fprintf(log, "\n");
	}

	fprintf(log, "||| Filter group delta tracing finished.\n|||\n");

	free(fDelta);
}

cFilterGroup::~cFilterGroup(){

	for(int i = 0; i < Input.size(); i++){
	
		if(Input[i] != NULL){
		
			Input[i] = NULL;
		}
	}
	Input.clear();
	for( int i = 0; i < Output.size(); i++){
	
		if(Output[i] != NULL){
		
			delete Output[i];
			Output[i] = NULL;
		}
	}
	Output.clear();
	for( int i = 0; i < Filters.size(); i++){
	
		if(Filters[i] != NULL){
		
			delete Filters[i];
			Filters[i] = NULL;
		}
	}
	Filters.clear();
	for(int i = 0; i< DevOutput.size(); i++){
	
		if(DevOutput[i] != NULL){
		
			delete DevOutput[i];
			DevOutput[i] = NULL;
		}
	}
	DevOutput.clear();

	if(DevInPtr.get() != NULL){
	
		cudaFree(DevInPtr.get());
	}

	if(DevOutPtr.get() != NULL){
	
		cudaFree(DevOutPtr.get());
	}

	if(fDevOffsets != NULL){
	
		cudaFree(fDevOffsets);
	}
	if(fOffsets != NULL){
	
		free(fOffsets);
	}

	if(fDeltaTemp != NULL){
	
		cudaFree(fDeltaTemp);
	}
	/*if(DevIn != NULL){
	
		delete DevIn;
		DevIn = NULL;
	}

	if(DevOut != NULL){
	
		delete DevOut;
		DevOut = NULL;
	}*/
}