
#include "cNetColumn.h"

void cNetColumn::Initialize()
{
	cLayer* LowerLayer;

	nLayerCount = GlobalValues.sLayerCount;
	nConvLayerCount = GlobalValues.sConvLayerCount;
	nPoolLayerCount = GlobalValues.sPoolLayerCount;
	nFullLayerCount = GlobalValues.sFullConnLayerCount;
	nFullNeuronCount = GlobalValues.sFullConnNeuronCount;

	FilterCounts = GlobalValues.FilterCounts;
	Layers = thrust::host_vector<cLayer*>();

	DevInputImage = NULL;

	cudaStreamCreate(&H2DStream);
	cudaStreamCreate(&D2HStream);
	cudaStreamCreate(&CalcStream);
	cudaStreamCreate(&D2DStream);
	 
	for(int i = 0 ; i < nLayerCount; i++){

		cLayer* TempLayer;

		TempLayer = new cLayer();

		TempLayer->H2DStream = &H2DStream;
		TempLayer->D2HStream = &D2HStream;
		TempLayer->CalcStream = &CalcStream;
		TempLayer->D2DStream = &D2DStream;

		if(i == 0){
		     
			// for the initialization of first layer we attached the input
			TempLayer->Input.push_back(InputImage);
		}

		TempLayer->Type = GlobalValues.LayerTypes[i];
		if(i != nLayerCount - 1){
		
			if(TempLayer->Type == cGlobalValues::CONVOLUTION){
			
				TempLayer->bCombineInput = false;
				TempLayer->bCombineOutput = false;

			}else if(TempLayer->Type == cGlobalValues::MAXPOOLING){
			
				TempLayer->bCombineInput = true;

				if(i != GlobalValues.sConvLayerCount + GlobalValues.sPoolLayerCount - 1)
					TempLayer->bCombineOutput = false;
				else TempLayer->bCombineOutput = true;

			}else{
			
				// for fully and output layers always combine input and output
				TempLayer->bCombineInput = true;
				TempLayer->bCombineOutput = true;
			}
		}

		if(i != 0){
			// hook the layer to the lower one
			Layers.back()->HigherLayer = (TempLayer);
			TempLayer->LowerLayer = (Layers.back());

		}else{
		
			TempLayer->LowerLayer = NULL;
		}

		Layers.push_back(TempLayer);

		Layers.back()->Initialize(i);

	}
}

// The parameter is the identify number of this column in the whole network
void cNetColumn::Trace(short sNumber){

	fprintf(logFile, "| Starting tracing No.%d net column:\n", sNumber);
	fprintf(logFile, "| Layer count: %d\n|\n", Layers.size());
	for(int i = 0; i < Layers.size(); i++){
	
		Layers[i]->Trace();
	}
	fprintf(logFile, "| Finishing tracig No.%d net column.\n|\n", sNumber);
}

cPixel* cNetColumn::ComputeImage(cImages* Input){

	InputImage = Input;
	if(DevInputImage != NULL) delete DevInputImage;
	DevInputImage = new cImagesGPU(*InputImage, true);

	for(int i = 0; i < Layers.size(); i++){
		if(i == 0) {

			if(Layers[i]->FilterGroup->DevInput.empty()) Layers[i]->FilterGroup->DevInput.push_back(DevInputImage);
			else Layers[i]->FilterGroup->CopyDevInput( DevInputImage, DevInputImage->size());
		}
		
		// allocate the space of input or output here?
		Layers[i]->Compute();
		
		checkCudaErrors(cudaEventSynchronize(Layers[i]->EventBusy));
		//if(GlobalValues.bUseCPUMem) Layers[i]->ClearGPUOutput();
	}

	return GetResult();
}

void cNetColumn::Clean(){

	for(int i = 0; i < Layers.size(); i++){
	
		Layers[i]->FilterGroup->Clean();
	}
}

void cNetColumn::Train(cImages* Target){

	TargetImage = Target;
	cImagesGPU* DevTarget = NULL;

	// back progation part to calcutate delta of each level
	for(int i = Layers.size()-1; i >= 0; i-- ){

		if(i == (Layers.size() - 1)){

			DevTarget = new cImagesGPU( *TargetImage, true);
			Layers[i]->SetTarget(DevTarget);
			//Layers[i]->TraceDelta();
		}

		if(i != 0) {

			Layers[i]->CalcDelta();
			//Layers[i-1]->TraceDelta();
		}
		
		checkCudaErrors(cudaEventRecord( Layers[i]->EventBusy, CalcStream));
		checkCudaErrors(cudaStreamWaitEvent( CalcStream, Layers[i]->EventBusy, 0));
	}

	// Weights update here, according to delta
	for(int i = 0; i < Layers.size(); i++){
	
		Layers[i]->Train();
	}
}

void cNetColumn::CopyWeightsD2H(){

	for(int i = 0; i < Layers.size(); i++){
	
		Layers[i]->CopyWeightsD2H();
	}
}

void cNetColumn::SaveWeights(){

	for(int i = 0; i < Layers.size(); i++){
	
		Layers[i]->SaveWeights();
	}
}

void cNetColumn::LoadWeights(){

	for(int i = 0; i < Layers.size(); i++){
	
		Layers[i]->LoadWeights();
	}
}

void cNetColumn::TraceDevOutput(int col){

	for(int i = 0; i < Layers.size(); i++){
	
		Layers[i]->TraceDevOutput(col);
	}
}

// Get the label value at the end of calculation
cPixel* cNetColumn::GetResult(){

	int L = Layers.size() - 1;

	cPixel* cTemp = NULL;

	cudaMallocHost((void**)&cTemp, GlobalValues.sOutputCount * sizeof(cPixel));
	checkCudaErrors(cudaMemcpy(cTemp, Layers[L]->FilterGroup->DevOutput[0]->data(), GlobalValues.sOutputCount * sizeof(cPixel), cudaMemcpyDeviceToHost));

	return cTemp;
}

void cNetColumn::TraceMaxPooling(){

	for( int i = 0; i < Layers.size(); i++){
	
		if(Layers[i]->Type == cGlobalValues::MAXPOOLING){
		
			Layers[i]->CopyWeightsD2H();
			Layers[i]->Trace();
		}
	}
}

cNetColumn::~cNetColumn(){

	if(InputImage != NULL) {

		delete InputImage;
		InputImage = NULL;
	}
	if(DevInputImage != NULL) {

		delete DevInputImage;
		DevInputImage = NULL;
	}

	for(int i = 0; i < nLayerCount; i++){
	
		if(Layers[i] != NULL)delete Layers[i];
		Layers[i] = NULL;
	}
	Layers.clear();

	if(TargetImage != NULL){

		delete TargetImage;
		TargetImage = NULL;
	}
}