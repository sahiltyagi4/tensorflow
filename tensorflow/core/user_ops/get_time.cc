#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <iostream>
#include <chrono>

using namespace tensorflow;
using namespace std::chrono;

REGISTER_OP("GetTime")
		.Input("input_loss_or_gradvars: float32")
		.Output("out_timestamp: float32");

class GetTimeOp : public OpKernel {
	public:
  		explicit GetTimeOp(OpKernelConstruction* context) : OpKernel(context) {}

  		void Compute(OpKernelContext* context) override {
  			//fetching the input tensor
  			const Tensor& input_tensor = context->input(0);
  			auto input = input_tensor.flat<float32>();

  			//creating the output tensor of dim (0)
  			Tensor* output_tensor = NULL;
  			OP_REQUIRES_OK(context, context->allocate_output(0, [], &output_tensor));
  			auto output = output_tensor->template flat<float32>();

  			microseconds ms = duration_cast< microseconds >(system_clock::now().time_since_epoch());
  			output_tensor = ms.count();
  			//output_tensor(0) = ms.count();
  		}
};

REGISTER_KERNEL_BUILDER(Name("GetTime").Device(DEVICE_CPU), GetTimeOp);