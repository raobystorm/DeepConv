
#include "cLoadFile.h"
#include "string.h"

// Global values here
int trainCount = 0;
int testCount = 0;
int imgWidth = 0;
int imgHeight = 0;
int magicNum = 0;

FILE* imgSet = NULL;
FILE* logFile = NULL;
FILE* imgLabel = NULL;
FILE* tstSet = NULL;
FILE* tstLabel = NULL;
FILE* weightsFile = NULL;

void cLoadFile::Initialize(){

	int readCount = 0;
	int readTmp = 0;

	imgSet = fopen(TRAINING_SET_IMAGE_PATH, "rb");
	tstSet = fopen(TESTING_SET_IMAGE_PATH, "rb");
	logFile = fopen(LOG_FILE_PATH, "w");
	imgLabel = fopen(TRAINING_SET_LABLE_PATH, "rb");
	tstLabel = fopen(TESTING_SET_LABLE_PATH, "rb");

	if(NULL == imgSet){
	
			throw "Image set open error.\n";
	}

	if(NULL == imgLabel){
	
			throw "Image label open error.\n";
	}

	if(NULL == logFile){
	
			printf("log file open error.\n");
			return;
	}

	if(NULL == tstSet){
	
			printf("test file open error.\n");
			return;
	}

	if(NULL == tstLabel){
	
			printf("test label open error.\n");
			return;
	}

	readCount = fread(&magicNum, sizeof(int), 1, imgSet);
	if(readCount == 0) throw "magic num read error.\n";
	// Transefer bigdian to littledian
	magicNum = BigtoLittle32(magicNum);

	readCount = fread(&trainCount, sizeof(int), 1, imgSet);
	if(readCount == 0) throw "training set size read error.\n";
	trainCount = BigtoLittle32(trainCount);

	readCount = fread(&readTmp, sizeof(int), 1, imgLabel);
	readCount = fread(&readTmp, sizeof(int), 1, imgLabel);
	readTmp = BigtoLittle32(readTmp);

	readCount = fread(&imgHeight, sizeof(int), 1, imgSet);
	if(readCount == 0) throw "image height read error.\n";
	imgHeight = BigtoLittle32(imgHeight);

	readCount = fread(&imgWidth, sizeof(int), 1, imgSet);
	if(readCount == 0) throw "image width size read error.\n";
	imgWidth = BigtoLittle32(imgWidth);

	readCount = fread(&readTmp, sizeof(int), 1, tstSet);

	readCount = fread(&testCount, sizeof(int), 1, tstSet);
	if(readCount == 0) throw "training set size read error.\n";
	testCount = BigtoLittle32(testCount);

	readCount = fread(&readTmp, sizeof(int), 1, tstLabel);
	readCount = fread(&readTmp, sizeof(int), 1, tstLabel);

	readCount = fread(&readTmp, sizeof(int), 1, tstSet);

	readCount = fread(&readTmp, sizeof(int), 1, tstSet);

}

cLoadFile::~cLoadFile(){

	if(NULL != logFile){
	
		fclose(logFile);
		logFile = NULL;
	}
	if(NULL != imgSet){
	
		fclose(imgSet);
		imgSet = NULL;
	}
	if(NULL != imgLabel){
	
		fclose(imgLabel);
		imgLabel = NULL;
	}
	if(NULL != tstSet){
	
		fclose(tstSet);
		tstSet = NULL;
	}
	if(NULL != tstLabel){
	
		fclose(tstLabel);
		tstLabel = NULL;
	}
}