#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <iostream>
#include <chrono>

using namespace tensorflow;
using namespace std::chrono;

REGISTER_OP("GetTime")
		.Input("input_loss_or_gradvars: int32")
		.Output("out_timestamp: int32");

class GetTimeOp : public OpKernel {
	public:
  		explicit GetTimeOp(OpKernelConstruction* context) : OpKernel(context) {}

  		void Compute(OpKernelContext* context) override {
  			//fetching the input tensor
  			const Tensor& input_tensor = context->input(0);
  			auto input = input_tensor.flat<int32>();

  			//creating the output tensor of dim (0)
  			Tensor* output_tensor = NULL;
  			OP_REQUIRES_OK(context, context->allocate_output(0, [], &output_tensor));
  			auto output = output_tensor->template flat<int32>();

  			microseconds ms = duration_cast< microseconds >(system_clock::now().time_since_epoch());
  			output(0) = (int32) ms.count();
  		}
};

REGISTER_KERNEL_BUILDER(Name("GetTime").Device(DEVICE_CPU), GetTimeOp);