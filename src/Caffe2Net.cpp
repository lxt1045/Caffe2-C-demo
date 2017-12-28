#ifdef WITH_CUDA
#include <caffe2/core/context_gpu.h>
#endif

#include "Caffe2Net.h"
#include <cstring>
/*
Caffe2Net::Caffe2Net(string initNet, string predictNet)
	: workspace(nullptr)
{
#ifdef WITH_CUDA
	DeviceOption option;
	option.set_device_type(CUDA);
	new CUDAContext(option);
#endif
	//载入部署模型
	NetDef init_net_def, predict_net_def;
	CAFFE_ENFORCE(ReadProtoFromFile(initNet, &init_net_def));
	CAFFE_ENFORCE(ReadProtoFromFile(predictNet, &predict_net_def));
#ifdef WITH_CUDA
	init_net_def.mutable_device_option()->set_device_type(CUDA);
	predict_net_def.mutable_device_option()->set_device_type(CUDA);
#else
	init_net_def.mutable_device_option()->set_device_type(CPU);
	predict_net_def.mutable_device_option()->set_device_type(CPU);
#endif
	//网络初始化
	workspace.RunNetOnce(init_net_def);
	//创建判别器
	predict_net = CreateNet(predict_net_def, &workspace);
}
//*/
Caffe2Net::Caffe2Net(string initNet, string predictNet)
	: workspace(nullptr)
{
#ifdef WITH_CUDA
	DeviceOption option;
	option.set_device_type(CUDA);
	new CUDAContext(option);
#endif

	//在这里增加自己的代码，以便支持 .pb 模型参数
	// pbtxt OR .pb
	std::string fileSuffix(".pb");
	if (initNet.compare(initNet.length() - fileSuffix.length(), fileSuffix.length(), fileSuffix) == 0)
	{
	}

	//载入部署模型
	NetDef init_net_def, predict_net_def;
	CAFFE_ENFORCE(ReadProtoFromFile(initNet, &init_net_def));
	CAFFE_ENFORCE(ReadProtoFromFile(predictNet, &predict_net_def));
#ifdef WITH_CUDA
	init_net_def.mutable_device_option()->set_device_type(CUDA);
	predict_net_def.mutable_device_option()->set_device_type(CUDA);
#else
	init_net_def.mutable_device_option()->set_device_type(CPU);
	predict_net_def.mutable_device_option()->set_device_type(CPU);	
#endif
	//网络初始化
	workspace.RunNetOnce(init_net_def);
	//创建判别器
	predict_net = CreateNet(predict_net_def,&workspace);
}

Caffe2Net::~Caffe2Net()
{
}

vector<float> Caffe2Net::predict(Mat img)
{
	//create input blob
#ifdef WITH_CUDA
	TensorCUDA input = TensorCUDA(preProcess(img));
	auto tensor = workspace.CreateBlob("data")->GetMutable<TensorCUDA>();
#else
	TensorCPU input = preProcess(img);
	auto tensor = workspace.CreateBlob("data")->GetMutable<TensorCPU>();
#endif
	tensor->ResizeLike(input);
	tensor->ShareData(input);
	//predict
	predict_net->Run();
	//get output blob
#ifdef WITH_CUDA
	TensorCPU output = TensorCPU(workspace.GetBlob("softmax")->Get<TensorCUDA>());
#else
	TensorCPU output = TensorCPU(workspace.GetBlob("softmax")->Get<TensorCPU>());
#endif
	return postProcess(output);
}

TensorCPU Caffe2Net::preProcess(Mat img)
{
}

vector<float> Caffe2Net::postProcess(TensorCPU output)
{
}
