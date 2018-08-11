#include <algorithm>
#include <cfloat>
#include "../filler.hpp"
#include "./depthwise_conv_layer.hpp"
//#include "caffe/util/math_functions.hpp"

namespace caffe {


	/*2017.11.10 
	Author:SyGoing(YangShu)
	*/
	//Depthwise Forward Convolution Kernel Function For GPU version
	template <typename Dtype>
	__global__ void DethwiseForwardGPUkernel(
		const Dtype *input,const int num_input,const int in_width,const int in_height,
		const Dtype *kernel,const int kernel_w,const int kernel_h,const int stride,
		const int pad,const int out_width,const int out_height,const int num_output,
	    Dtype *output, const int outputs, const Dtype* const bias, const bool bias_term_){

		int thread_id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
		if (thread_id >= outputs) return;

		//计算当前output像素点的四维索引索引
		const int width= thread_id % out_width;
		const int height = (thread_id / out_width) % out_height;
		const int channel = (thread_id / out_width / out_height) % num_output;
		const int batchID = thread_id / out_width / out_height / num_output;//batch size

		const int in_d =channel;

		const int input_offset_temp = (batchID * num_input + in_d) * (in_height * in_width);//当前output channel对应的input channel 的指针

		const int input_height_start = height * stride - pad;
		const int input_width_start = width* stride - pad;
		const int input_height_end = input_height_start + kernel_h;
		const int input_width_end = input_width_start + kernel_w;

		float sum = 0;
		if (input_height_start >= 0 && input_width_start >= 0 &&
			input_height_end < in_height && input_width_end < in_width)
		{
            #pragma unroll
			for (int f_r = 0; f_r < kernel_h; ++f_r) {
				const int in_r = input_height_start + f_r;
				#pragma unroll
				for (int f_c = 0; f_c < kernel_w; ++f_c) {
					const int in_c = input_width_start + f_c;

					const int input_offset =
						(input_offset_temp)+(in_r * in_width) + in_c;
					const int filter_offset = f_c + kernel_w * f_r + channel*kernel_w*kernel_h;
					sum += (*(input + input_offset)) * (*(kernel + filter_offset));
				}
			}
		}
		else {
			#pragma unroll
			for (int f_r = 0; f_r < kernel_h; ++f_r) {
				const int in_r = input_height_start + f_r;
				#pragma unroll
				for (int f_c = 0; f_c < kernel_w; ++f_c) {
					const int in_c = input_width_start + f_c;

					if (in_r >= 0 && in_r < in_height && in_c >= 0 && in_c < in_width) {
						const int in_c = input_width_start + f_c;

						const int input_offset =
							(input_offset_temp)+(in_r * in_width) + in_c;

						const int filter_offset = f_c + kernel_w * f_r + channel*kernel_w*kernel_h;
						sum += (*(input + input_offset)) * (*(kernel + filter_offset));
					}
				}
			}
		}

		//是否有偏置
		if (bias_term_) {
			sum += bias[channel];
		}

		output[thread_id] = sum;
	}

	//2017.11.10 SyGoing Add
	//overload the Forward_gpu For DepthwiseLayer
	void DepthwiseConvLayer::Forward_gpu(
		const vector<Blob*>& bottom, const vector<Blob*>& top) {
		const real_t* weight = this->blobs_[0]->gpu_data();

		int* kernel_shape_data = this->kernel_shape_.mutable_cpu_data();
		int* stride_data = this->stride_.mutable_cpu_data();
		int* pad_data = this->pad_.mutable_cpu_data();

		for (int i = 0; i < bottom.size(); ++i) {
			const real_t* bottom_data = bottom[i]->gpu_data();
			real_t* top_data = top[i]->mutable_gpu_data();
			const int count = top[i]->count();
			vector<int> shape_ = bottom[i]->shape();
			const int channels_ = shape_[1];
			const int height_ = shape_[2];
			const int width_ = shape_[3];


			//这一块是用的数据拷贝
			const int kernel_h_ = kernel_shape_data[0];
			const int kernel_w_ = kernel_shape_data[1];
			const int stride = stride_data[0];
			const int pad = pad_data[0];

			const int conved_height = this->output_shape_[0];
			const int conved_width = this->output_shape_[1];

			const bool bias_term_ = this->bias_term_;


			

			if (bias_term_) {
				const real_t* const bias = this->blobs_[1]->gpu_data();
				DethwiseForwardGPUkernel<real_t> << <SCAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(
					bottom_data, channels_, width_, height_, weight, kernel_h_, kernel_w_, stride,
					pad, conved_width, conved_height, channels_,top_data, count, bias, bias_term_);
			}
			else {
				DethwiseForwardGPUkernel<real_t> << <SCAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(
					bottom_data, channels_, width_, height_, weight, kernel_h_, kernel_w_, stride,
					pad, conved_width, conved_height, channels_,top_data, count, 0, bias_term_);
			}

			//2017.11.10 注释不用
			/*stupid method 
			   
			*/
			//for (int n = 0; n < this->num_; ++n) {
			//	for (int c = 0; c < this->channels_; ++c){
			//		const Dtype* const bottom_slice = bottom_data + (n *  this->channels_ + c) * bottom[i]->shape()[2] * bottom[i]->shape()[3];
			//		const Dtype* const weight_slice = weight + c * kernel_shape_data[0] * kernel_shape_data[1];
			//		Dtype*  top_slice = top_data + (n *  this->channels_ + c) * this->output_shape_[0] * this->output_shape_[1];
			//		
			//		//2017.11.08  
			//		this->mforward_gpu_gemm(bottom_slice, weight_slice, top_slice);
			//	}
			//	if (this->bias_term_) {
			//		const Dtype* bias = this->blobs_[1]->gpu_data();
			//		this->mforward_gpu_bias(top_data + n * this->top_dim_, bias);
			//	}
			//}
		}
	}
}  // namespace caffe
