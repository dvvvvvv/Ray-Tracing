#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "VecMath.h"

struct Screen
{
	float w;	// horizontal length
	float h;	// vertical length

	float min_x;	// left most point
	float max_y;	// top point
	float z;	// dist from eye



	int resolution_w;
	int resolution_h;

	float pixel_w;	// width of a pixel (w/resolution_w)
	float pixel_h;	// height of a pixel (h/resolution_h)
};


struct Ray
{
	float3 p0;	//	starting point (Eye)
	float3 v;	// Normalized Direction
};


struct Material
{
	float kd;
	float3 diffuseIntensity;
	float ks;
	float3 specularIntensity;
	int exponent;
	float kr;
};

struct Polygon
{
	float3 vertex1;
	float3 vertex2;
	float3 vertex3;

	Material material;
};


struct Sphere
{
	float3 p;	// center position
	float r;	// radius
	Material material;
};


__device__ float3 GetNormal(Sphere &s, float3 Intersection)
{
	return Normalize(Sub(Intersection, s.p));
}

__device__ float3 GetNormal(Polygon &p, float3 Intersection,float3 eye)
{
	float3 normal = Normalize(Cross(Sub(p.vertex1, p.vertex2), Sub(p.vertex1, p.vertex3)));
	float3 toEye = Normalize(Sub(eye, Intersection));

	if (Dot(normal, toEye)<0) normal = Scale(-1, normal);

	return normal;
}



__device__ float Area(Polygon &p)
{
	float area =
		Length(
		Cross(
		Sub(p.vertex2, p.vertex1),
		Sub(p.vertex3, p.vertex1)
		)
		) / 2;

	return area;
}

__device__ float Area(float3 v1, float3 v2, float3 v3)
{
	float area =
		Length(
		Cross(
		Sub(v2, v1),
		Sub(v3, v1)
		)
		) / 2;

	return area;
}