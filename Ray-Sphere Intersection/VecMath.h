#pragma once

#include <math.h>
#ifndef VEC_MATH
#define VEC_MATH

__device__ float3 Add(float3 &a, float3 &b)
{
	float3 out;
	out.x = a.x + b.x;
	out.y = a.y + b.y;
	out.z = a.z + b.z;

	return out;
}

__device__ float3 Sub(float3 &a, float3 &b)
{
	float3 out;
	out.x = a.x - b.x;
	out.y = a.y - b.y;
	out.z = a.z - b.z;

	return out;
}

__device__ float3 Scale(float s, float3 &a)
{
	float3 out;
	out.x = s*a.x;
	out.y = s*a.y;
	out.z = s*a.z;

	return out;
}


__device__ float Dot(float3 &a, float3 &b)
{
	float out = a.x*b.x + a.y*b.y + a.z*b.z;

	return out;
}


__device__ float3 Cross(float3 &a, float3 &b)
{
	float3 out;

	out.x = a.y*b.z-a.z*b.y;
	out.y = a.x*b.z-a.z*b.x;
	out.z = a.x*b.y-a.y*b.x;

	return out;
}

__device__ float Length(float3 &a)
{
	return sqrt( a.x*a.x+a.y*a.y+a.z*a.z );
}

__device__ float Length2(float3 &a)
{
	return a.x*a.x+a.y*a.y+a.z*a.z;
}

__device__ float3 Normalize(float3 &a)
{
	
	return Scale(1.f/Length(a), a);
}


#endif