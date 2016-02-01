#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "VecMath.h"
#include "Struct.h"

class Object
{
public:
	Object();
	virtual ~Object();

	Material material;

	virtual __device__ float3 GetNormal(float3 intersection);
	virtual __device__ float3 GetIntersection(Ray r, float t);
	float3 GetDiffuse(Ray r, float3 light);
	float3 Get
};

