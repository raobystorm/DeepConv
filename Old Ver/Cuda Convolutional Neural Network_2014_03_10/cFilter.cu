
#include "cFilter.h"

void cFilter::Initialize(short Level){

	sLevel = Level;

	fWeights = NULL;
	fDevWeights = NULL;

	nWeights = NULL;
	nDevWeights = NULL;

	bWeights = NULL;
	bDevWeights = NULL;

	Pitch = 0;

	if(Type == cGlobalValues::MAXPOOLING){

		bWeights = (bool*)malloc(MapSize.height * MapSize.width * sizeof(bool));
		checkCudaErrors(cudaMalloc( (bool**)&bDevWeights, MapSize.height * MapSize.width * sizeof(bool)));

		memset(bWeights, false, MapSize.height * MapSize.width * sizeof(bool));
		checkCudaErrors(cudaMemset(bDevWeights, false, MapSize.height * MapSize.width * sizeof(bool)));
		
		// When it's a poolig layer, we can return now
		return;
	}

	if(GlobalValues.bFloatWeight){

		NppiSize weightSize, offsetSize;

		if(Type == cGlobalValues::CONVOLUTION){

			weightSize.height = kerSize.height;
			weightSize.width = kerSize.width;

			offsetSize.height = 1;
			offsetSize.width = 1;

		}else{
			
			weightSize.height = MapSize.height;
			weightSize.width = MapSize.width;

			offsetSize.height = 1;
			offsetSize.width = GlobalValues.sFullConnNeuronCount;
			if(Type == cGlobalValues::OUTPUT) offsetSize.width = 0;
		}

		fWeights = (Npp32f*)malloc(weightSize.height * weightSize.width * sizeof(Npp32f));
		checkCudaErrors(cudaMalloc((Npp32f**)&fDevWeights, weightSize.width * sizeof(Npp32f) * weightSize.height));

		// Randomize the weights before going on
		if(Type == cGlobalValues::FULLYCONNECTED){

			for(int i = 0; i < (weightSize.height * weightSize.width); i++){

				fWeights[i] =  fRandom(GlobalValues.fMinRandWeight, GlobalValues.fMaxRandWeight); //1.0 / 1800; 5;*//*fRanGauss(-5.0f, 5.0f, 0.0f, 0.5f) / 360;
			}
		}else if(Type == cGlobalValues::CONVOLUTION){

			for(int i = 0; i < (weightSize.height * weightSize.width); i++){

				fWeights[i] = fRanGauss(-5.,5.,0.0,0.5)/weightSize.height;//fRanSigmoid(-5.0f, 5.0f, 3.0f)/(weightSize.width);
			}
		}else{

			for(int i = 0; i < (weightSize.height * weightSize.width); i++){

				fWeights[i] = fRandom(GlobalValues.fMinRandWeight, GlobalValues.fMaxRandWeight);//*//*fRandom(GlobalValues.fMinRandWeight, GlobalValues.fMaxRandWeight);
			}
		}
		checkCudaErrors(cudaMemcpy( fDevWeights, fWeights, weightSize.width * sizeof(Npp32f) * weightSize.height, cudaMemcpyHostToDevice));
	}
	else{ 
		// $unfinished!
		nWeights = (Npp32s*) malloc(kerSize.width * kerSize.height * sizeof(Npp32s));

		for(int i = 0; i < (kerSize.width * kerSize.height); i++){
			nWeights[i] = nRandom(GlobalValues.nMinRandWeight, GlobalValues.nMaxRandWeight);
		}
	}
}

void cFilter::Compute(){


}

void cFilter::Train(){


}

void cFilter::Trace(short sNumber){

	fprintf(logFile, "|||| Start tracing No.%d filter.\n",sNumber);
	if(Type == cGlobalValues::CONVOLUTION){

		fprintf(logFile, "|||| kernel size - width: %d height: %d\n", kerSize.width, kerSize.height);

		if(GlobalValues.bFloatWeight){
			// float type weight here. integer type below
			for(int i = 0; i < kerSize.height; i++){
		
				for(int j = 0; j < kerSize.width; j++){
			
					fprintf(logFile, "%f ", fWeights[i*kerSize.width + j]);
				}
				fprintf(logFile, "\n");
			}
			fprintf(logFile, "\n|||| Filter tracing finished.\n||||\n");

		}else{
		
			for(int i = 0; i < MapSize.height; i++){
		
				for(int j = 0; j < MapSize.width; j++){
			
					fprintf(logFile, "%d ", nWeights[i*MapSize.width + j]);
				}
				fprintf(logFile, "\n");
			}
			fprintf(logFile, "\n|||| Filter tracing finished.\n||||\n");
		}

	}else if(Type == cGlobalValues::MAXPOOLING){
	
		fprintf(logFile, "|||| Map size - width: %d, height: %d.\n", MapSize.width, MapSize.height);
		for(int i = 0; i < MapSize.height; i++){
			for(int j = 0; j < MapSize.width; j++){
			
				if(bWeights[i * MapSize.width + j] == false) fprintf( logFile, "%d ", 0);
				else fprintf( logFile, "%d ", 1);
			}
			fprintf(logFile, "\n");
		}
		fprintf(logFile, "|||| Filter tracing finished.\n||||\n");
	}else if(Type == cGlobalValues::FULLYCONNECTED){
	
		fprintf(logFile, "|||| Map size - width: %d, height: %d.\n", MapSize.width, MapSize.height);
		for(int i = 0; i < MapSize.height; i++){
		
				for(int j = 0; j < MapSize.width; j++){
			
					fprintf(logFile, "%f ", fWeights[i*MapSize.width + j]);
				}
				fprintf(logFile, "\n");
			}
			fprintf(logFile, "\n|||| Filter tracing finished.\n||||\n");
	}else{

		fprintf(logFile, "|||| Map size - width: %d, height: %d.\n", MapSize.width, MapSize.height);
		for(int i = 0; i < MapSize.height; i++){
		
				for(int j = 0; j < MapSize.width; j++){
			
					fprintf(logFile, "%f ", fWeights[i*MapSize.width + j]);
				}
				fprintf(logFile, "\n");
			}
			fprintf(logFile, "\n|||| Filter tracing finished.\n||||\n");
	}
}

void cFilter::CopyWeightsH2D(){

	if(Type == cGlobalValues::MAXPOOLING){

		checkCudaErrors(cudaMemcpy( bDevWeights, bWeights,
			MapSize.height * MapSize.width * sizeof(bool), cudaMemcpyHostToDevice));

	}else if(GlobalValues.bFloatWeight){
	
		if(fDevWeights == NULL){
		
			checkCudaErrors(cudaMalloc((Npp32f**)&fDevWeights, kerSize.height * kerSize.width * sizeof(Npp32f)));
		}

		checkCudaErrors(cudaMemcpy( fDevWeights, fWeights,
			kerSize.height * kerSize.width * sizeof(Npp32f), cudaMemcpyHostToDevice));

	}else{
	
		if(nDevWeights == NULL){
		
			checkCudaErrors(cudaMalloc((Npp32s**)&nDevWeights, kerSize.height * kerSize.width * sizeof(Npp32s)));
		}

		checkCudaErrors(cudaMemcpyAsync( nDevWeights, nWeights,
			kerSize.height * kerSize.width * sizeof(Npp32s), cudaMemcpyHostToDevice));
	}
}

void cFilter::CopyWeightsD2H(){

	if(Type == cGlobalValues::MAXPOOLING){
	
		checkCudaErrors(cudaMemcpy( bWeights, bDevWeights, MapSize.height * MapSize.width * sizeof(bool), cudaMemcpyDeviceToHost));

	}else if(GlobalValues.bFloatWeight){
	
		if(Type == cGlobalValues::CONVOLUTION)	checkCudaErrors(cudaMemcpy( fDevWeights, fWeights, kerSize.width * sizeof(Npp32f) * kerSize.height, cudaMemcpyHostToDevice));
		else checkCudaErrors(cudaMemcpy( fWeights, fDevWeights, MapSize.height * MapSize.width * sizeof(Npp32f), cudaMemcpyDeviceToHost));

	}else{
	
		if(Type == cGlobalValues::CONVOLUTION)	checkCudaErrors(cudaMemcpy( nDevWeights, nWeights, kerSize.width * sizeof(Npp32s) * kerSize.height, cudaMemcpyHostToDevice));
		else checkCudaErrors(cudaMemcpy( nWeights, nDevWeights, MapSize.height * MapSize.width * sizeof(Npp32s), cudaMemcpyDeviceToHost));
	}
}

cFilter::~cFilter(){

	if(fWeights != NULL){
	
		free(fWeights);
		fWeights = NULL;
	}
	if(nWeights != NULL){
	
		free(nWeights);
		nWeights = NULL;
	}
	if(bWeights != NULL){
	
		free(bWeights);
		bWeights = NULL;
	}
	if(fDevWeights != NULL){
	
		cudaFree(fDevWeights);
		fDevWeights = NULL;
	}
	if(nDevWeights != NULL){
	
		cudaFree(nDevWeights);
		nDevWeights = NULL;
	}
	if(bDevWeights != NULL){
	
		cudaFree(fDevWeights);
		bDevWeights = NULL;
	}
}