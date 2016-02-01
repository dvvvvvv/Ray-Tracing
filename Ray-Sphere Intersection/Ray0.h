#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

void BuildImage(uchar3* h_color_buffer, int w, int h);