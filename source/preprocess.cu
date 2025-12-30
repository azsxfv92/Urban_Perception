#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <algorithm>

__global__ void preprocess_kernel(
    const unsigned char* src, //input : pinned memory
    float* dst, //output : GPU memory
    int src_w, int src_h, // original image shape
    int dst_w, int dst_h  // target shape
){
    // get thread coordinate information
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // ignore oversized thread
    if (x >= dst_w || y >= dst_h) return;

    float scale_x = (float)src_w / dst_w;
    float scale_y = (float)src_h / dst_h;

    float src_x = (x + 0.5f) * scale_x - 0.5f;
    float src_y = (y + 0.5f) * scale_y - 0.5f;
    
    int x_low = (int)floorf(src_x);
    int y_low = (int)floorf(src_y);

    int x_high = min(x_low + 1, src_w -1);
    int y_high = min(y_low + 1, src_h -1 );

    x_low = max(x_low, 0);
    y_low = max(y_low, 0);

    float dx = src_x - x_low;
    float dy = src_y - y_low;

    float w_tl = (1.0f - dx) * (1.0f - dy);
    float w_tr = dx * (1.0f - dy);
    float w_bl = (1.0f - dx) * dy;
    float w_br = dx * dy;

    // length of original images(byte) * channel(3)
    int src_stride = src_w * 3;
    // area of target images
    int dst_stride = dst_w * dst_h;

    for (int c=0; c<3; c++){
        int src_c = 2-c;

        int idx_tl = (y_low * src_stride + x_low * 3) + src_c;
        int idx_tr = (y_low * src_stride + x_high * 3) + src_c;
        int idx_bl = (y_high * src_stride + x_low * 3) + src_c;
        int idx_br = (y_high * src_stride + x_high * 3) + src_c;

        float val = 
            (float)src[idx_tl] * w_tl +
            (float)src[idx_tr] * w_tr +
            (float)src[idx_bl] * w_bl +
            (float)src[idx_br] * w_br;

            dst[c * dst_stride + y * dst_w + x] = val/255.0f;
    }
}

void cuda_preprocess(
    void* src_ptr,
    void* dst_ptr,
    int src_w, int src_h,
    int dst_w, int dst_h,
    cudaStream_t stream
){
    dim3 block(16,16);

    dim3 grid(
        (dst_w + block.x -1) / block.x,
        (dst_h + block.y -1) / block.y
    );

    preprocess_kernel<<<grid, block, 0, stream>>>(
        (unsigned char*)src_ptr,
        (float*)dst_ptr,
        src_w, src_h,
        dst_w, dst_h
    );
}
