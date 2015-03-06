

#include "cLayer.h"

#include <ImageIO.h>

//------------------------------------------------------------------------
//-------------------------Host Funcitons Below-----------------------
//------------------------------------------------------------------------

// According to the type of the layer
// Initialize the instance
void cLayer::Initialize(short sLevel){

	cImages* TempOut;

	nLayerLevel = sLevel;

	bFirstCom = true;

	nFilterCount = GlobalValues.FilterMultipliers[nLayerLevel];

	nFilterGroupCount = 1;

	Input = thrust::host_vector<cImages*>();
	DevInput = thrust::host_vector<cImagesGPU*>();
	Output = thrust::host_vector<cImages*>();
	DevOutput = thrust::host_vector<cImagesGPU*>();

	cudaEventCreate(&EventBusy, cudaEventBlockingSync);

	// Attach input of the lower layer
	if(sLevel != 0){

		Input = LowerLayer->Output;
		DevInput = LowerLayer->DevOutput;
	}

	InSize.width = Input[0]->width();
	InSize.height = Input[0]->height();

	FilterGroup = new cFilterGroup();

	if(Type == cGlobalValues::CONVOLUTION){
	
		kerSize = GlobalValues.kerSizes[(nLayerLevel)];

		FilterGroup->Type = cGlobalValues::CONVOLUTION;

	}else if(Type == cGlobalValues::FULLYCONNECTED){
	
		// Only one group in fully connected layer
		FilterGroup->Type =cGlobalValues::FULLYCONNECTED;

	}else if(Type == cGlobalValues::MAXPOOLING){
	
		// One group in maxpooling, handle all the input together
		FilterGroup->Type = cGlobalValues::MAXPOOLING;
		// For maxpooling layer there is no need to calculate delta value
		// Directly using the lower layer delta value is OK.

	}else if(Type == cGlobalValues::OUTPUT){
		
		FilterGroup->Type = cGlobalValues::OUTPUT;
	}
	
	FilterGroup->Input = Input;
	FilterGroup->DevInput = DevInput;
	FilterGroup->InSize = InSize;

	FilterGroup->H2DStream = H2DStream;
	FilterGroup->D2HStream = D2HStream;
	FilterGroup->CalcStream = CalcStream;
	FilterGroup->D2DStream = D2DStream;

	FilterGroup->Initialize(nLayerLevel);

	for(int j = 0; j < FilterGroup->Output.size(); j++){
		
		Output.push_back(FilterGroup->Output[j]);
		DevOutput.push_back(FilterGroup->DevOutput[j]);
	}

	OutSize.height = Output[0]->height();
	OutSize.width = Output[0]->width();
}

void cLayer::Trace(){

	fprintf(logFile, "|| Starting tracing No.%d layer:\n", nLayerLevel);
	if(Type == cGlobalValues::CONVOLUTION){
	
		fprintf(logFile, "|| Layer type: convolution layer.\n");
	}else if(Type == cGlobalValues::MAXPOOLING){
	
		fprintf(logFile, "|| Layer type: max pooling layer.\n");
	}else if(Type == cGlobalValues::FULLYCONNECTED){
	
		fprintf(logFile, "|| Layer type: fully-connected layer.\n");
	}else if(Type == cGlobalValues::OUTPUT){
	
		fprintf(logFile, "|| Layer type: output layer.\n");
	}

	fprintf(logFile, "|| Filter group count:%d.\n", 1);
	
	FilterGroup->Trace(1);
	fprintf(logFile, "|| No.%d Layer tracing finished.\n||\n", nLayerLevel);
}

void cLayer::TraceDelta(){

	FILE* log = NULL;
	char* cpath, *num;
	char d[] = "_Delta";
	cpath = (char*)malloc(255 * sizeof(char));
	memset( cpath, 0, 255 * sizeof(char));
	num = (char*)malloc(10 * sizeof(char));

	itoa(nLayerLevel, num, 10);
	strcat( cpath, LOG_FILE_BATCH_PATH);
	strcat( cpath, d);
	strcat( cpath, num);
	strcat( cpath, ".txt");

	log = fopen( cpath, "w");

	fprintf(log, "|| Starting tracing delta of No.%d layer:\n", nLayerLevel);
	if(Type == cGlobalValues::CONVOLUTION){
	
		fprintf(log, "|| Layer type: convolution layer.\n");
	}else if(Type == cGlobalValues::MAXPOOLING){
	
		fprintf(log, "|| Layer type: max pooling layer.\n");
	}else if(Type == cGlobalValues::FULLYCONNECTED){
	
		fprintf(log, "|| Layer type: fully-connected layer.\n");
	}else if(Type == cGlobalValues::OUTPUT){
	
		fprintf(log, "|| Layer type: output layer.\n");
	}

	fprintf(log, "|| Filter group count:%d.\n", 1);
	
	FilterGroup->TraceDelta(log);
	fprintf(log, "|| No.%d Layer tracing finished.\n||\n", nLayerLevel);

	fclose(log);
}

void cLayer::SetTarget(cImagesGPU* Target){

	if(Type != cGlobalValues::OUTPUT) return;

	FilterGroup->SetTarget( Target);
}

void cLayer::CalcDelta(){

	NppiSize iSize = LowerLayer->FilterGroup->OutSize;

	FilterGroup->CalcDelta(LowerLayer->FilterGroup->fDevDelta, iSize);
}

void cLayer::Train(){

	if(Type == cGlobalValues::MAXPOOLING) return;

	if(nLayerLevel != 0) FilterGroup->Train(LowerLayer->FilterGroup->OutSize.width * LowerLayer->FilterGroup->OutSize.height);
	else FilterGroup->Train(FilterGroup->MapSize.width*FilterGroup->MapSize.height);
}

// Allocate the combined all-in-one memory block for GPU
void cLayer::AllocDevIn(){

	if(!bCombineInput) return;

	if(DevIn != NULL) cudaFree(DevIn);

	int size = 0;

	if(GlobalValues.bUseCPUMem){
	
		size = (Input.size() * (Input[0]->width()) * (Input[0]->height()));
	}else{
	
		size = (DevInput.size() * (DevInput[0]->width()) * (DevInput[0]->height()));
	}

	checkCudaErrors(cudaMalloc((float**)&DevIn, size * sizeof(float)));
}

// Check all the flags and layer attributes to decide the allocation of device memory
// Including the device input & output of the layer and the device input & output of the filter groups
// For every layer and every goup, DevIn, DevOut, DevInput and DevOutput must be allocated
void cLayer::CheckDeviceMemory(){

	// DevInput of the layers
	/*if(nLayerLevel != 0) DevInput = LowerLayer->DevOutput;
	else{
	
		DevInput.push_back(new cImagesGPU(*Input[0]));
	}

	for(int i = 0; i < FilterGroups.size(); i++){
		// DevInput of filter groups
		if((FilterGroups[i]->DevInput.empty())||GlobalValues.bUseCPUMem){

			FilterGroups[i]->DevInput.clear();
			for(int j = 0; j < Input.size(); j++){
		
				FilterGroups[i]->DevInput.push_back(DevInput[j]);
			}
			FilterGroups[i]->InputCount = Input.size();
		}
		// DevOutput of the group
		if(FilterGroups[i]->DevOutput.empty()||GlobalValues.bUseCPUMem){

			FilterGroups[i]->ClearGPUOutput();
			FilterGroups[i]->DevOutputAlloc();
		}
	}*/
	/*if(Type == cGlobalValues::CONVOLUTION){
		// Convolution dont use DevIn & DevOut
		// DevInput of Layer
		if(nLayerLevel != 0){
		
			DevInput = LowerLayer->DevOutput;

		}else{

			cImagesGPU* DevSrc = new cImagesGPU(*Input[0]);
			DevInput.push_back( DevSrc);
		}
			
		FilterGroup->DevInput.push_back(DevInput[0]);
		FilterGroup->DevOutputAlloc();

	}else if(Type == cGlobalValues::MAXPOOLING){
		// DevInput layer and group
		DevInput = LowerLayer->DevOutput;
		FilterGroup->DevInput = DevInput;
		// DevIn layer and Group
		AllocDevIn();
		FilterGroup->DevIn = DevIn;
		// DevOut group
		FilterGroup->DevOutAlloc();
		// DevOut of layer & DevOutput of group
		if(bCombineOutput){
		
			DevOut = FilterGroup->DevOut;

		}else{
		
			DevOut = NULL;
			FilterGroup->DevOutputAlloc();
		}

	}else if(Type == cGlobalValues::FULLYCONNECTED){
		// DevInput of layer and group
		DevInput = LowerLayer->DevOutput;
		FilterGroup->DevInput = DevInput;
		// DevIn of layer and group
		if(LowerLayer->bCombineOutput){ 

			DevIn = LowerLayer->DevOut;

		}else{
		
			AllocDevIn();
		}
		FilterGroup->DevIn = DevIn;

		if(bCombineOutput){
			// DevOut of group and layer, not using output here
			FilterGroup->DevOutAlloc();
			DevOut = FilterGroup->DevOut;
		}else{
		
			// Usually flly connected layers only output combined results
			// Not over here
			FilterGroup->DevOutputAlloc();
		}

	}else if(Type == cGlobalValues::OUTPUT){
		// Usually dont use devinput
		DevInput = LowerLayer->DevOutput;
		FilterGroup->DevInput = DevInput;

		DevIn = LowerLayer->DevOut;
		FilterGroup->DevIn = DevIn;

		FilterGroup->DevOutAlloc();
		DevOut = FilterGroup->DevOut;

		// Because output is always need to copy back to CPU
		// Use DevOutput
		FilterGroup->DevOutputAlloc();
		DevOutput = FilterGroup->DevOutput;
	}

	// For all layers adds filter groups' devoutput to layer's output
	// When its not empty and we are using GPU mem directly dont alloc new ones
	if(DevOutput.empty()||GlobalValues.bUseCPUMem){
		DevOutput.clear();
		for(int j = 0; j < FilterGroup->DevOutput.size(); j++)
			DevOutput.push_back(FilterGroup->DevOutput[j]);
	}*/
}

void cLayer::Compute(){

	// Mainly allocate the memory for device computation
	// CheckDeviceMemory();

	// Using CPU mem and copy inputs to device
	//if(nLayerLevel == 0) FilterGroup->CopyInputs();
	
	// Setting the pointers of DevInput into one device_ptr
	if(Type != cGlobalValues::MAXPOOLING){

		if(bFirstCom){

			if(nLayerLevel == GlobalValues.sConvLayerCount + GlobalValues.sPoolLayerCount)
				FilterGroup->ExtendInput1D();

			else FilterGroup->ExtendInput();

		}else if(nLayerLevel != 0){

			if(nLayerLevel == GlobalValues.sConvLayerCount + GlobalValues.sPoolLayerCount)
				FilterGroup->CopyDevInput1D(LowerLayer->FilterGroup->DevOutput[0], LowerLayer->FilterGroup->DevOutput[0]->size());

			else FilterGroup->CopyDevInput(LowerLayer->FilterGroup->DevOutput[0], LowerLayer->FilterGroup->DevOutput[0]->size());
		}
	}

	FilterGroup->Compute(0);
	
	cudaEventRecord( EventBusy, *CalcStream);

	/*checkCudaErrors( cudaEventSynchronize( EventBusy));*/

	/*if(GlobalValues.bUseCPUMem)	FilterGroup->CopyResults();

	cudaEventRecord( EventBusy, GlobalValues.D2HStream);

	checkCudaErrors( cudaEventSynchronize( EventBusy));

	TraceOutput();*/

	/*if(Type == cGlobalValues::FULLYCONNECTED){
		
		FilterGroups[0]->ClearGPUTmp();
	}*/

	// Waiting for the GPU asynchronize computation finish
	
	if(bFirstCom == true) bFirstCom = false;
}

void cLayer::CopyWeightsD2H(){

	FilterGroup->CopyWeightsD2H();
	//checkCudaErrors(cudaEventSynchronize(FilterGroup->EventD2H));
}

void cLayer::SaveWeights(){

	FilterGroup->SaveWeights();
}

void cLayer::LoadWeights(){

	FilterGroup->LoadWeights();
}

void cLayer::ClearGPUOutput(){

	if(GlobalValues.bUseCPUMem){

		FilterGroup->ClearGPUOutput();
	}
}

void cLayer::TraceDevOutput(int col){

	FilterGroup->TraceDevOutput(col);
}

cLayer::~cLayer(){

	for(int i = 0; i < nFilterGroupCount; i++){
	
		if(FilterGroup!= NULL) {

			delete FilterGroup;
			FilterGroup = NULL;
		}
	}
	for(int i = 0; i < Input.size(); i++){
	
		if(Input[i] != NULL){

			Input[i] = NULL;
		}
	}
	Input.clear();
	for(int i = 0; i < Output.size(); i++){
	
		if(Output[i] != NULL){
		
			Output[i] = NULL;
		}
	}
	Output.clear();

	if(DevIn != NULL){
	
		delete DevIn;
		DevIn = NULL;
	}

	if(DevOut != NULL){
	
		delete DevOut;
		DevOut = NULL;
	}
}