
#include "windows.h"
#include "cNetwork.h"

void cNetwork::Initialize(){

	cNetColumn* TempColumn;

	// re-calculate the parameters above if neccessary
	CalculateScale();

	// Working here decide to get the first image from set or not
	InputImage = cImages(imgWidth, imgHeight);
	TargetImage = cImages(GlobalValues.sOutputCount, 1);

	ColParam = thrust::host_vector<cColParameter>(GlobalValues.sNetColumnCount);
	ColEvents = (HANDLE*)malloc(NetColumn.size() * sizeof(HANDLE));

	for(int i = 0; i < GlobalValues.sNetColumnCount; i++){

		ColEvents[i] = CreateEvent( NULL, true, false, NULL);

		ColParam[i].Network = this;
		ColParam[i].Event = ColEvents[i];
		ColParam[i].label = 0;
		ColParam[i].number = i;
		ColParam[i].Input = NULL;
		ColParam[i].Target = NULL;

		TempColumn = new cNetColumn();

		TempColumn->InputImage = &InputImage;

		TempColumn->Initialize();

		NetColumn.push_back(TempColumn);
	}
}

DWORD WINAPI ComputeColumn(LPVOID lpParameter){

	cColParameter* cParam = (cColParameter*) lpParameter; 

	cNetwork* Network = (cNetwork*)cParam->Network;

	cParam->Output = Network->NetColumn[cParam->number]->ComputeImage( cParam->Input);

	SetEvent(cParam->Event);

	return 0;
}

DWORD WINAPI TrainColumn(LPVOID lpParameter){

	cColParameter* cParam = (cColParameter*) lpParameter;

	cNetwork* Network = (cNetwork*)cParam->Network;

	Network->NetColumn[cParam->number]->Train( cParam->Target);

	return 0;
}

bool cNetwork::Train(cImages* Input, int label, int Count){

	/*HANDLE handle;
	HRESULT hResult;

	for(int i = 0; i < NetColumn.size(); i++){

		if(ResetEvent(ColEvents[i])) hResult = GetLastError();

		ColParam[i].Input = Input;
		ColParam[i].label = label;
	
		CloseHandle(CreateThread( NULL, 0, ComputeColumn, &ColParam[i], 0, NULL));
	}

	WaitForMultipleObjects( NetColumn.size(), ColEvents, true, INFINITE);

	int l = -1;

	l = GetResult();

	

	for(int i = 0; i < NetColumn.size(); i++){
	
		ColParam[i].Target = &TargetImage;

		CloseHandle(CreateThread( NULL, 0, TrainColumn, &ColParam[i], 0, NULL));
	}*/

	//Trace();

	int l = Compute(Input);

	//checkCudaErrors(cudaDeviceSynchronize());

	if((Count == 15000)){

		Trace();

		TraceDevOutput();
	
	}

	// Set the correct output of this input
	for(int i = 0; i < GlobalValues.sOutputCount; i++){

		if(i == label) *(TargetImage.data(i)) = 255;
		else *(TargetImage.data(i)) = 0;
	}

	// Train all the columns of the same image here
	// When training mode is single training mode
	if(GlobalValues.TrainMode == cGlobalValues::SINGLE){

		for(int i = 0; i < GlobalValues.sNetColumnCount; i++){
	
			NetColumn[i]->Train(&TargetImage);
		}
	}

	checkCudaErrors(cudaDeviceSynchronize());

	if(l == label) return true;
	else return false;
}

void cNetwork::Trace(){

	CopyWeightsD2H();
	
	if(logFile == NULL){
	
		printf("Log file is closed.\n\n");
		system("pause");
	}

	fprintf(logFile, "Starting tracing convolutional neural network...\n\n");
	
	fprintf(logFile, "Net column count: %d\n\n", NetColumn.size());

	for(int i = 0; i < NetColumn.size(); i++){
	
		NetColumn[i]->Trace(i+1);
	}

	fprintf(logFile, "Network tracing finish.\n");
}

void cNetwork::TraceDevOutput(){

	for(int i = 0; i < NetColumn.size(); i++){
	
		NetColumn[i]->TraceDevOutput(i);
	}
}

// for one picture, compute the output
cPixel* cNetwork::Compute(cImages* Input, int col){
	
	int* res = NULL;
	res = (int*)malloc(GlobalValues.sOutputCount * sizeof(int));
	memset(res, 0, GlobalValues.sOutputCount * sizeof(int));
	cPixel* cTemp = NULL;

	cTemp = NetColumn[col]->ComputeImage(Input);

	return cTemp;
}

int cNetwork::GetResult(){

	int* res = NULL;
	res = (int*)malloc(GlobalValues.sOutputCount * sizeof(int));
	memset(res, 0, GlobalValues.sOutputCount * sizeof(int));
	int temp = 0, max = 0;

	for(int i = 0; i < NetColumn.size(); i++){

		// Add up the results of all columns
		for(int j = 0; j < GlobalValues.sOutputCount; j++){
			
			temp = int(ColParam[i].Output[j]);
			res[j] += temp;
		}

		cudaFreeHost(ColParam[i].Output);
	}

	for(int i = 0; i < GlobalValues.sOutputCount; i++){
	
		if(res[i] > max){
		
			max = res[i];
			temp = i;
		}
	}

	free(res);
	return temp;
}

// for one picture, compute the output
int cNetwork::Compute(cImages* Input){
	
	int* res = NULL;
	res = (int*)malloc(GlobalValues.sOutputCount * sizeof(int));
	memset(res, 0, GlobalValues.sOutputCount * sizeof(int));
	cPixel* cTemp = NULL;
	int temp = 0, max = 0;

	// Compute all the columns with the same image
	if(GlobalValues.TrainMode == cGlobalValues::eTrainMode::SINGLE){
	
		for(int i = 0; i < NetColumn.size(); i++){
		
			cTemp = NetColumn[i]->ComputeImage(Input);

			// Add up the results of all columns
			for(int j = 0; j < GlobalValues.sOutputCount; j++){
			
				temp = int(cTemp[j]) / NetColumn.size();
				res[j] += temp;
			}

			cudaFreeHost(cTemp);
		}
		// Do some prepare works here to insure the result of network
	}

	for(int i = 0; i < GlobalValues.sOutputCount; i++){
	
		if(res[i] > max){
		
			max = res[i];
			temp = i;
		}
	}

	return temp;
}

void cNetwork::SaveWeights(){

	weightsFile = fopen(WEIGHTS_SAVE_LOAD_PATH, "wb");

	if(weightsFile == NULL){
	
		printf("\nWeights file load error!\n");
		return;
	}

	for(int i = 0; i < NetColumn.size(); i++){
	
		NetColumn[i]->CopyWeightsD2H();
	}

	for(int i = 0; i < NetColumn.size(); i++){
	
		NetColumn[i]->SaveWeights();
	}

	fclose(weightsFile);
	weightsFile = NULL;
}

void cNetwork::LoadWeights(){

	weightsFile = fopen(WEIGHTS_SAVE_LOAD_PATH, "rb");

	if(weightsFile == NULL){
	
		printf("\nWeights file load error!\n");
		return;
	}

	for(int i = 0; i < NetColumn.size(); i++){
	
		NetColumn[i]->LoadWeights();
	}

	fclose(weightsFile);
	weightsFile = NULL;
}

void cNetwork::CopyWeightsD2H(){

	for(int i = 0; i < NetColumn.size(); i++){
	
		NetColumn[i]->CopyWeightsD2H();
	}
}

// Re-compute the scale of the model current input and configuration
void cNetwork::CalculateScale(){


}

void cNetwork::TrainSet(){


}

void cNetwork::ComputeSet(){


}

cNetwork::~cNetwork(){

	for(int i = 0; i < GlobalValues.sNetColumnCount; i++){
	
		if(NetColumn[i] != NULL){
		
			delete NetColumn[i];
			NetColumn[i] = NULL;
		}
	}
	NetColumn.clear();
}