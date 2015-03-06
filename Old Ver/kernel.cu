
#include "npp.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cNetwork.h"
#include "math.h"

#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>

cNetwork* Network;
cLoadFile ConFiles;

#define TEST_COUNT							1
#define KERNEL_WIDTH						5
#define KERNEL_HEIGHT						5

#define POOLING_WIDTH						2
#define POOLING_HEIGHT						2
/*
thrust::device_vector<Npp8u*> testDevSrc = thrust::device_vector<Npp8u*>();
thrust::device_vector<Npp8u*> testDevDst = thrust::device_vector<Npp8u*>();

thrust::host_vector<Npp8u*> testHostSrc = thrust::host_vector<Npp8u*>();
thrust::host_vector<Npp8u*> testHostDst = thrust::host_vector<Npp8u*>();
*/
thrust::host_vector<cImagesGPU*> testDevSrc = thrust::host_vector<cImagesGPU*>();
thrust::host_vector<cImagesGPU*> testDevDst = thrust::host_vector<cImagesGPU*>();

thrust::host_vector<cImages*> testHostSrc = thrust::host_vector<cImages*>();
thrust::host_vector<cImages*> testHostDst = thrust::host_vector<cImages*>();


/*inline __global__ void maxpooling(Npp8u* DevSrc, Npp8u* DevDst, NppiSize SrcSize, NppiSize DstSize){

	// if(blockIdx.x >= testDevSrc.size()) return;
	// Work here...Fix pooling kernel problem

	int max = 0;
	int Idx_1 = 0;

	for(int i = 0; i < POOLING_WIDTH; i++){

		if(threadIdx.x * POOLING_WIDTH + i >= SrcSize.width) return;
		for(int j = 0; j < POOLING_HEIGHT; j++){
		
			if(threadIdx.y * POOLING_HEIGHT + j >= SrcSize.height) return;

			Idx_1 = (threadIdx.y * POOLING_HEIGHT + j) * SrcSize.width + (threadIdx.x * POOLING_WIDTH + i);
			if(max < DevSrc[Idx_1]) max = DevSrc[Idx_1];
		}
	}
	DevDst[threadIdx.y * DstSize.width + threadIdx.x] = max;
}*/

inline void maxpoolingCPU(Npp8u* HostSrc, Npp8u* HostDst, NppiSize SrcSize, NppiSize DstSize){

	int xSrc, ySrc, max, Idx_1;
	for(int i = 0; i < DstSize.width * DstSize.height; i++){
		
		xSrc = (i%DstSize.width) * POOLING_WIDTH;
		ySrc = (i/DstSize.width) * POOLING_HEIGHT;
		max = 0;

		for(int j = 0; j < POOLING_WIDTH; j++){
		
			if(xSrc + j >= SrcSize.width) break;
			for(int k = 0; k < POOLING_HEIGHT; k++){
			
				if(ySrc + k >= SrcSize.height) break;

				Idx_1 = (ySrc + k) * SrcSize.width + xSrc + j;
				if(max < HostSrc[Idx_1]) max = HostSrc[Idx_1];
			}
		}

		HostDst[i] = max;
	}
}

inline void ConvolutionCPU(Npp8u* Src, Npp8u* Dst, float* fKernel, NppiSize ROISize, NppiSize kerSize){

	for(int i = 0; i < ROISize.height - KERNEL_HEIGHT + 1; i ++){
		for(int j = 0; j < ROISize.width - KERNEL_WIDTH + 1; j++){
		
			int res = 0;

			for(int k = 0; k < KERNEL_HEIGHT; k++){
				if(i + k >= ROISize.height) break;
				for(int l = 0; l < KERNEL_WIDTH; l++){

					if(j + l >= ROISize.width) break;
					res += Src[(i + k) * ROISize.width + j + l] * fKernel[k * kerSize.width + l];
				}
			}

			Dst[i * ROISize.width + j] = res;
		}
	}
}

inline int cudaDeviceInit(int argc, const char **argv)
{
    int deviceCount;
    checkCudaErrors(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0)
    {
        std::cerr << "CUDA error: no devices supporting CUDA." << std::endl;
        exit(EXIT_FAILURE);
    }

    int dev = findCudaDevice(argc, argv);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    std::cerr << "cudaSetDevice GPU" << dev << " = " << deviceProp.name << std::endl;

    checkCudaErrors(cudaSetDevice(dev));

    return dev;
}

int main(int argc, char *argv[])
{
	int label = 0;
	char temp = 0;
	int err = 0;
	int readCount =0;
	NppiSize kerSize = {0, 0};
	NppiSize SrcSize = {0, 0};
	dim3 threadsPerBlock(0, 0);
	cudaStream_t Stream_1;
	cudaEvent_t Event_1;

	cImages* oHostSrc = NULL;
	cImages* oHostDst = NULL;
	cImages* oDstCPU = NULL;

	cImagesGPU* DevTemp = NULL;
	cImagesGPU* DevDst = NULL;

	Npp8u* HostTemp = NULL;
	
	cImagesGPU* DevSrc = NULL;

	StopWatchInterface* hTimer = NULL;
	float Time;

	void* tmp;

	try{

		NppGpuComputeCapability computeCap = nppGetGpuComputeCapability();

		if (computeCap < NPP_CUDA_2_0)
		{
			std::cerr << "This sample needs a GPU with Compute Capability 2.0 or higher" << std::endl;
			return EXIT_SUCCESS;
		 }
		
		cudaDeviceReset();

		cudaDeviceInit(argc, (const char **)argv);
		checkCudaErrors(cudaDeviceReset());

		ConFiles.Initialize();
		fprintf( logFile,"Loading file successed! Starting initialize network...\n");

		GlobalValues.Initialize();

		Network = new cNetwork;

		Network->Initialize();

		oHostSrc = new cImages(imgWidth, imgHeight);

		//system("pause");
		
		//tmp = malloc(sizeof(Npp8u) * imgWidth * imgHeight);
		
		//----------------------------------------------------------------------------------------
		//---------------------Main Working Part Below--------------------------------------------
		//----------------------------------------------------------------------------------------

		std::string filePath = std::string();
		
		char* cpath;
		cpath = (char*)malloc(20 * sizeof(char));
	
		itoa(1, cpath, 10);
	
		filePath = OUTPUT_SAVE_PATH + std::string("_Level") + std::string(cpath);

		int r = 0, w = 0;
		float rate = 0.0;

		//Network->LoadWeights();
		cudaDeviceSynchronize();

		for(int i = 0; i < trainCount; i++){
		
			readCount = fread(oHostSrc->data(), sizeof(cPixel), imgWidth * imgHeight, imgSet);
			fread(&temp, sizeof(char), 1, imgLabel);

			label = temp;

			if(Network->Train(oHostSrc, label, i)){
			
				r++;
			}else{
			
				w++;
			}

			if(i % 500 == 499){
				
				rate = r * 1.0 / (r + w);
				fprintf(logFile, "O:%f	", rate);
				printf("O:%f	", rate);
			}

			if(i % 2500 == 2499) {
				fprintf(logFile, "\n");printf("\n");
			}

			if(i%GlobalValues.sBatchSize == (GlobalValues.sBatchSize - 1)) GlobalValues.fLearnMomentum *= 0.92f;
		}

		r = 0; 
		w = 0;
		rate = 0.0;
		//Network->SaveWeights();

		/*for(int i = 0; i < testCount; i++){
		
			readCount = fread(oHostSrc->data(), sizeof(cPixel), imgWidth * imgHeight, tstSet);
			fread(&temp, sizeof(char), 1, tstLabel);

			label = temp;

			if(label == Network->Compute(oHostSrc)){
			
				r++;
				
			}else{
			
				w++;
			}
		}

		rate = r * 1.0 / (r + w);
		printf("\n\nTest correct rate: %f.\n", rate);
		fprintf(logFile, "\n\nTest correct rate: %f.\n", rate);*/

		//----------------------------------------------------------------------------------------
		//-------------------------Main Working Part Above----------------------------------------
		//----------------------------------------------------------------------------------------
		 
		//memcpy( tmp, tmp, sizeof(Npp8u) * imgWidth * imgHeight);
		
		//Network->TraceOutput();
		
		//Network->Compute(oHostSrc);

		//Network->Trace();

		/*
		cudaStreamCreate(&Stream_1);
		cudaEventCreate(&Event_1);
		nppSetStream( Stream_1);

		tmp = malloc(sizeof(Npp8u) * imgWidth * imgHeight);

		std::string filePath;
		char* cpath;

		cpath = (char*)malloc(sizeof(char) * 10);
		testDevSrc.clear();
		testDevDst.clear();
		testHostSrc.clear();
		testHostDst.clear();

		for(int i = 0; i < TEST_COUNT ; i++){
		
			readCount = fread(tmp, sizeof(Npp8u) , imgWidth * imgHeight, imgSet);

			if(readCount != imgHeight * imgWidth) {
		
				err = ferror(imgSet);
				throw;
			}

			oHostSrc = new cImages(imgWidth, imgHeight);
			memcpy( oHostSrc->data(), tmp, sizeof(Npp8u) * imgWidth * imgHeight);
			testHostSrc.push_back(oHostSrc);

			DevSrc = new cImagesGPU(imgWidth, imgHeight);
			checkCudaErrors(cudaMemcpy( DevSrc->data(), tmp, imgWidth * imgHeight * sizeof(Npp8u), cudaMemcpyHostToDevice));
			testDevSrc.push_back(DevSrc);

			oHostDst = new cImages(imgWidth - KERNEL_HEIGHT + 1, imgHeight - KERNEL_HEIGHT + 1);
			testHostDst.push_back(oHostDst);

			DevDst = new cImagesGPU(imgWidth - KERNEL_HEIGHT + 1, imgHeight - KERNEL_HEIGHT + 1);
			testDevDst.push_back(DevDst);
		}

		float* fKernel = NULL;

		fKernel = (float*)malloc(KERNEL_WIDTH * KERNEL_HEIGHT * sizeof(float));

		printf("\n");
		for(int i = 0; i < KERNEL_HEIGHT; i++){
			for(int j =0; j < KERNEL_WIDTH; j++){

				fKernel[i * KERNEL_WIDTH + j] = fRandom(-0.05, 0.05);
				printf("%f ", fKernel[i * KERNEL_WIDTH + j]);
			}
			printf("\n");
		}
		printf("\n");

		SrcSize.height = imgHeight;
		SrcSize.width = imgWidth;

		kerSize.height = KERNEL_HEIGHT;
		kerSize.width = KERNEL_WIDTH;

		NppiPoint Anchor;
		Anchor.x = 1;
		Anchor.x = 1;

		sdkCreateTimer(&hTimer);
		sdkResetTimer(&hTimer);
		sdkStartTimer(&hTimer);

		for(int i = 0; i < TEST_COUNT; i++){
		
			nppiFilter32f_8u_C1R (testDevSrc[i]->data(), testDevSrc[i]->pitch(), testDevDst[i]->data(),
				testDevDst[i]->pitch(), SrcSize, fKernel, kerSize, Anchor);
		}

		//checkCudaErrors(cudaEventRecord(Event_1, Stream_1));

		cudaDeviceSynchronize();
		
		sdkStopTimer(&hTimer);
		Time = 0.0;
		Time = sdkGetTimerValue(&hTimer);
		
		printf("Complete max pooling using GPU for %d maps.\n", TEST_COUNT);
		printf("Finished in %f msecs.\n", Time);
		system("pause");

		sdkResetTimer(&hTimer);
		sdkStartTimer(&hTimer);

		for(int i = 0; i < TEST_COUNT; i++){
		
			ConvolutionCPU(testHostSrc[i]->data(), testHostDst[i]->data(), fKernel, SrcSize, kerSize);
		}
		
		sdkStopTimer(&hTimer);
		Time = 0.0;
		Time = sdkGetTimerValue(&hTimer);

		printf("Complete max pooling using CPU for %d maps.\n", TEST_COUNT);
		printf("Finished in %f msecs.\n", Time);
		system("pause");

		float Err = 0.0f;

		oHostDst = new cImages(imgWidth, imgHeight);
		DevDst[0].copyTo(oHostDst->data(), oHostDst->pitch());

		for(int i = 0; i < imgHeight - KERNEL_HEIGHT + 1; i++){
		
			for(int j = 0; j < imgWidth - KERNEL_WIDTH + 1; j++){
			
				Err += sqrt((float)(oHostDst->data()[i * (imgHeight - KERNEL_HEIGHT + 1) + j] - testHostDst[0]->data()[(imgHeight - KERNEL_HEIGHT + 1) + j]));
			}
		}

		printf("\nError rate: %f\n", Err);*/

		/*for(int i = 0; i < TEST_COUNT; i++){
		
			checkCudaErrors(cudaMemcpy(oHostDst->data(), testDevDst[i], 
				DstSize.width * DstSize.height * sizeof(Npp8u), cudaMemcpyDeviceToHost));

			itoa(i, cpath, 10);

			filePath = IMAGE_SAVE_PATH + std::string(cpath) + "GPU.pgm";

			npp::saveImage(filePath, *oHostDst);*/

			/*memcpy(oDstCPU->data(), testHostDst[i], DstSize.width * DstSize.height * sizeof(Npp8u));

			filePath = IMAGE_SAVE_PATH + std::string(cpath) + "CPU.pgm";

			npp::saveImage(filePath, *oDstCPU);
			
		}

		/*printf("Reading image success.\n");

		printf("Saving image...\n\n");

		printf("good\n");*/

		for(int i = 0; i < testDevSrc.size(); i++){
		
			if(testDevSrc[i] != NULL){
			
				cudaFree(testDevSrc[i]);
				testDevSrc[i] = NULL;
			}
		}
		testDevSrc.clear();

		for(int i = 0; i < testDevDst.size(); i++){
		
			if(testDevDst[i] != NULL){
			
				cudaFree(testDevDst[i]);
				testDevDst[i] = NULL;
			}
		}
		testDevDst.clear();

		for(int i = 0; i < testHostSrc.size(); i++){
		
			if(testHostSrc[i] != NULL){
			
				cudaFree(testHostSrc[i]);
				testHostSrc[i] = NULL;
			}
		}
		testHostSrc.clear();

		for(int i = 0; i < testHostDst.size(); i++){
		
			if(testHostDst[i] != NULL){
			
				cudaFree(testHostDst[i]);
				testHostDst[i] = NULL;
			}
		}
		testHostDst.clear();

		printf("Success!\n");
		//system("pause");

		cudaDeviceReset();
		exit(EXIT_SUCCESS);

	}
	catch(char* err){

		/*if(logFile != NULL){
		
			fprintf(logFile, err);
			fprintf(logFile, "\n");
			fclose(logFile);
			logFile = NULL;
		}

		if(imgSet != NULL){
		
			fclose(imgSet);
			imgSet = NULL;
		}
	
		printf("Some error occured.\n");
		system("pause");*/
	}
}