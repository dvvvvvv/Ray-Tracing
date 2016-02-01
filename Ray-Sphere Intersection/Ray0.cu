#pragma once

#include <stdio.h>
#include "Ray0.h"
#include "Struct.h"
#include "VecMath.h"

#define SPHERE_SIZE 2
#define POLYGON_SIZE 6
#define MAX_REFLECTION 10

__device__ bool GetIntersectionWithSphere(Ray &r, Sphere &s, float3 &intersection);
__device__ bool GetIntersectionWithPolygon(Ray &r, Polygon &p, float3 &intersection);
__device__ bool GetIntersection(Ray r, float3 &intersection, float3 &normal, Material &material);

__device__ float3 MakeColor(float3 vnormal, float3 intersection, Material material, float3 eye, int count);
__device__ float3 GetDiffuse(float3 normal, float3 light, Material material);
__device__ float3 GetSpecular(float3 toEye, float3 reflection, Material material);
__device__ float3 GetReflection(Ray r,Material material, int count);

__device__ Ray MakeRay(int pixel_x, int pixel_y);
__device__ Ray MakeRay(float3 p0, float3 v);
__device__ Polygon MakePolygon(float3 p1, float3 p2, float3 p3);
__device__ Sphere MakeSphere(float3 p, float r);

// Global Variables
__device__ float3 eye_p;
__device__ Screen screen;
__device__ Sphere sphere[SPHERE_SIZE];
__device__ Polygon poly[POLYGON_SIZE];
__device__ float3 light;
__device__ float3 lightIntensity;
__device__ float3 ambientLight;

__global__ void InitEnv(int r_w, int r_h)
{
	eye_p = make_float3(0.f, 0.f, -200.f);

	screen.w = 300.f;
	screen.h = 200.f;
	screen.min_x = -150.f;
	screen.max_y = 100.f;
	screen.z = 0.f;
	screen.resolution_w = r_w;
	screen.resolution_h = r_h;
	screen.pixel_w = screen.w / r_w;
	screen.pixel_h = screen.h / r_h;
	
	sphere[0] = MakeSphere(make_float3(0.f, -50.f, 350.f), 40.f);
	sphere[0].material.kd = 1;
	sphere[0].material.diffuseIntensity = make_float3(1.0f, 0.f, 0.f);
	sphere[0].material.ks = 0.5f;
	sphere[0].material.specularIntensity = make_float3(1.f, 1.f, 1.f);
	sphere[0].material.exponent = 64;
	sphere[0].material.kr = 0.5f;

	sphere[1] = MakeSphere(make_float3(20.f, -20.f, 275.f), 25.f);
	sphere[1].material.diffuseIntensity = make_float3(0.f, 1.f, 1.f);


	poly[0] = MakePolygon(
		make_float3(0.f, -100.f,400.f),
		make_float3(-100.f, -100.f,300.f),
		make_float3(100.f, -100.f,300.f)
		);
	poly[1] = MakePolygon(
		make_float3(-100.f, -100.f, 300.f),
		make_float3(100.f, -100.f, 300.f),
		make_float3(0.f, -100.f, 200.f)
		);
	poly[2] = MakePolygon(
		make_float3(0.f, 42.f, 400.f),
		make_float3(0.f, -100.f, 400.f),
		make_float3(-100.f, -100.f, 300.f)
		);
	poly[3] = MakePolygon(
		make_float3(0.f, 42.f, 400.f),
		make_float3(-100.f, 42.f, 300.f),
		make_float3(-100.f, -100.f, 300.f)
		);
	poly[4] = MakePolygon(
		make_float3(0.f, 42.f, 400.f),
		make_float3(0.f, -100.f, 400.f),
		make_float3(100.f, -100.f, 300.f)
		);
	poly[5] = MakePolygon(
		make_float3(0.f, 42.f, 400.f),
		make_float3(100.f, 42.f, 300.f),
		make_float3(100.f, -100.f, 300.f)
		);
	

	for (int i = 0; i < POLYGON_SIZE; i++){
		poly[i].material.kd = 0.1f;
		poly[i].material.diffuseIntensity = make_float3(0.f, 0.f, 0.f);
		poly[i].material.ks = 0.5f;
		poly[i].material.specularIntensity = make_float3(1.f, 1.f, 1.f);
		poly[i].material.exponent = 512;
		poly[i].material.kr = 1.f;
	}

	ambientLight = make_float3(0.1f, 0.1f, 0.1f);

	light = make_float3(100.f, 300.f, 0.f);
	lightIntensity = make_float3(0.8f, 0.8f, 0.8f);
}

__device__ bool GetIntersectionWithSphere(Ray &r, Sphere &s,float3 &intersection)
{
	float t;
	float3 vector2Sphere = Sub(r.p0, s.p);


	float b = Dot(Scale(2, r.v), vector2Sphere);
	float c = Length2(vector2Sphere) - (s.r * s.r);

	float det = (b*b) - (4 * c);

	if (det > 0)
	{
		t = (-b - sqrt(det)) / 2;
		
	}
	else if (det == 0)
	{
		t = -b / 2;
	}
	else
	{
		return false;
	}
	
	intersection = Add(r.p0, Scale(t, r.v));
	if (t < 0.01){
		return false;
	}
	else return true;
	
}

__device__ bool GetIntersectionWithPolygon(Ray &r, Polygon &p, float3 &intersection)
{
	float3 normal = GetNormal(p,p.vertex1,eye_p);

	float3 VP = Sub(p.vertex1, r.p0);
	
	float NdotV = Dot(normal, r.v);
	float NdotVP = Dot(normal, VP);

	if (NdotV == 0) return false;

	float t = NdotVP / NdotV;

	if (t < 0.01) return false;
	intersection = Add(r.p0, Scale(t, r.v));
	
	float area = Area(p);
	float alpha = Area(intersection, p.vertex1, p.vertex2) / area;
	float beta = Area(intersection, p.vertex1, p.vertex3) / area;
	float gamma = Area(intersection, p.vertex2, p.vertex3) / area;

	if (alpha >= 0 && alpha <= 1 &&
		beta >= 0 && beta <= 1 &&
		gamma >= 0 && gamma <=1 &&
		alpha + beta + gamma <=1.01)
	{
		return true;
	}
	
	return false;
}

__device__ bool GetIntersection(Ray r, float3 &intersection, float3 &normal, Material &material)
{
	bool hasIntersection = false;

	float minLength= 999999;

	float3 minNormal;
	Material minMaterial;
	float3 minIntersection;
	
	for (int i = 0; i < SPHERE_SIZE; i++)
	{
		float3 tmpIntersection;
		
		if (GetIntersectionWithSphere(r, sphere[i], tmpIntersection))
		{
			float thisLength = Length(Sub(r.p0, tmpIntersection));
			if (thisLength < minLength)
			{
				hasIntersection = true;
				minIntersection = tmpIntersection;
				minLength = thisLength;
				minNormal = GetNormal(sphere[i], tmpIntersection);
				minMaterial = sphere[i].material;
			}
		}		
	}
	
	
	for (int i = 0; i < POLYGON_SIZE; i++)
	{
		float3 tmpIntersection;

		if (GetIntersectionWithPolygon(r,poly[i],tmpIntersection))
		{
			float thisLength = Length(Sub(r.p0, tmpIntersection));
			if (thisLength < minLength)
			{
				hasIntersection = true;
				minIntersection = tmpIntersection;
				minLength = thisLength;
				minNormal = GetNormal(poly[i],tmpIntersection,eye_p);
				minMaterial = poly[i].material;
				

			}
		}
	}
	if (hasIntersection)
	{
		intersection = minIntersection;
		normal = minNormal;
		material = minMaterial;
	}

	return hasIntersection;
	
}

__device__ Ray MakeRay(int pixel_x, int pixel_y)
{
	float3 pixel_center;
	pixel_center.x = screen.min_x + pixel_x*screen.pixel_w + (screen.pixel_w/2.f);
	pixel_center.y = screen.max_y - pixel_y*screen.pixel_h + (screen.pixel_h/2.f);
	pixel_center.z = screen.z;

	Ray r;
	r.p0 = eye_p;

	r.v = Sub(pixel_center, r.p0);
	r.v = Normalize(r.v);

	return r;
}


__device__ Ray MakeRay(float3 p0, float3 v)
{
	Ray r;
	r.p0 = p0;
	r.v = v;

	return r;
}

__device__ Polygon MakePolygon(float3 p1, float3 p2, float3 p3)
{
	Polygon ret;

	ret.vertex1 = p1;
	ret.vertex2 = p2;
	ret.vertex3 = p3;

	ret.material.kd = 1.0f;
	ret.material.kr = 1.0f;
	ret.material.ks = 1.0f;
	ret.material.specularIntensity = make_float3(1.0f, 1.0f, 1.0f);
	ret.material.diffuseIntensity = make_float3(1.0f, 1.0f, 1.0f);
	ret.material.exponent = 64;

	return ret;
}

__device__ Sphere MakeSphere(float3 p, float r)
{
	Sphere ret;

	ret.p = p;
	ret.r = r;

	ret.material.kd = 1.0f;
	ret.material.kr = 1.0f;
	ret.material.ks = 1.0f;
	ret.material.specularIntensity = make_float3(1.0f, 1.0f, 1.0f);
	ret.material.diffuseIntensity = make_float3(1.0f, 1.0f, 1.0f);
	ret.material.exponent = 64;

	return ret;
}

__device__ float3 GetAmbient(Material material)
{
	float3 color;

	color.x = material.diffuseIntensity.x * ambientLight.x;
	color.y = material.diffuseIntensity.y * ambientLight.y;
	color.z = material.diffuseIntensity.z * ambientLight.z;

	return color;
}

__device__ float3 GetDiffuse(float3 normal, float3 light,Material material)
{
	float3 color;

	float NL = Dot(normal, light);

	if (NL < 0){
		NL = 0;
	}

	color.x = material.kd * material.diffuseIntensity.x * lightIntensity.x * NL;
	color.y = material.kd * material.diffuseIntensity.y * lightIntensity.y * NL;
	color.z = material.kd * material.diffuseIntensity.z * lightIntensity.z * NL;

	return color;
}

__device__ float3 GetSpecular(float3 toEye, float3 reflection, Material material)
{
	float3 color;

	float EVsqrt = Dot(toEye, reflection);
	float EV = pow(EVsqrt, material.exponent);

	if (EVsqrt > 0){
		color.x = material.specularIntensity.x * material.ks * lightIntensity.x * EV;
		color.y = material.specularIntensity.y * material.ks * lightIntensity.y * EV;
		color.z = material.specularIntensity.z * material.ks * lightIntensity.z * EV;
	}
	else
	{
		color = make_float3(0.f, 0.f, 0.f);
	}

	return color;
}

__device__ float3 GetReflection(Ray r, Material material, int count)
{
	float3 color = make_float3(0.f, 0.f, 0.f);
	//return color;
	Ray rRay = r;

	float3 rIntersection;
	float3 rNormal;
	Material rMaterial;
	float kr = material.kr;

	while (GetIntersection(rRay, rIntersection, rNormal, rMaterial) && count >0)
	{
		float3 r2Light = Normalize(Sub(light, rIntersection));
		float3 r2Eye = Scale(-1,rRay.v);
		float3 rReflection = Normalize(Sub(Scale(2 * Dot(r2Light, rNormal), rNormal), r2Light));
		

		float3 ambient = GetAmbient(rMaterial);
		float3 diffuse = GetDiffuse(rNormal, r2Light, rMaterial);
		float3 specular = GetSpecular(r2Eye, rReflection, rMaterial);

		Ray toLight = MakeRay(rIntersection, Normalize(r2Light));
		float3 dummyVector;
		Material dummyMaterial;

		float3 tmpColor = Scale(kr,ambient);
		
		if (!GetIntersection(toLight, dummyVector, dummyVector, dummyMaterial))
		{
			tmpColor = Add(tmpColor, Scale(kr, Add(diffuse, specular)));
		}

		color = Add(color, tmpColor);
		kr *= rMaterial.kr;

		float3 next = Normalize(Sub(Scale(2 * Dot(Scale(-1, rRay.v), rNormal), rNormal), Scale(-1, rRay.v)));
		rRay = MakeRay(rIntersection, next);		
		count--;
	}

	return color;
}

__device__ float3 MakeColor(float3 vnormal, float3 intersection, Material material, float3 eye, int count)
{
	float3 vlight = Normalize(Sub(light, intersection));
	float3 v2Eye = Normalize(Sub(eye, intersection));
	float3 vreflection = Normalize(Sub(Scale(2 * Dot(vlight, vnormal), vnormal), vlight));

	float3 ambient;
	float3 diffuse;
	float3 specular;
	float3 reflection;

	//calc ambient
	ambient = GetAmbient(material);

	//calc diffuse
	diffuse = GetDiffuse(vnormal, vlight, material);
	
	
	//calc specular
	specular = GetSpecular(v2Eye, vreflection, material);

	//calc reflection
	Ray rRay = MakeRay(intersection, Normalize(Sub(Scale(2 * Dot(v2Eye, vnormal), vnormal), v2Eye)));

	reflection = GetReflection(rRay, material, MAX_REFLECTION);
	
	//calc final color
	Ray toLight = MakeRay(intersection, vlight);
	float3 dummyVector;
	Material dummyMaterial;

	float3 color = Add(ambient,reflection);

	if (!GetIntersection(toLight, dummyVector, dummyVector, dummyMaterial))
	{
		color.x += diffuse.x + specular.x;
		color.y += diffuse.y + specular.y;
		color.z += diffuse.z + specular.z;
	}

	return color;
}


__global__ void RayTracing(uchar3* d_color_buffer)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int pixel_x_index = tid%screen.resolution_w;
	int pixel_y_index = tid/screen.resolution_w;
	
	if ( tid < screen.resolution_w*screen.resolution_h )
	{
		
		float3 color;  // Result color


		// Make a ray through the pixal of (pixel_x_index, pixel_y_index)
		Ray ray = MakeRay(pixel_x_index, pixel_y_index);

		// Test if intersection between the ray and sphere0 is happened.
		
		float3 intersection;
		float3 normal;
		Material material;

		if (GetIntersection(ray, intersection, normal, material))
		{
			color = MakeColor(normal, intersection, material, eye_p, MAX_REFLECTION);
		}
		else
		{
			// Background Color
			color = make_float3(0.f, 0.f, 0.f);
		}
		



		// Scale up the color values to 0~255 and Copy the result color
		// R
		int tmp = round(color.x*255);
		if ( tmp < 0 ) tmp = 0;
		if ( tmp > 255 ) tmp = 255;
		d_color_buffer[tid].x = tmp;

		// G
		tmp = round(color.y*255);
		if ( tmp < 0 ) tmp = 0;
		if ( tmp > 255 ) tmp = 255;
		d_color_buffer[tid].y = tmp;

		// B
		tmp = round(color.z*255);
		if ( tmp < 0 ) tmp = 0;
		if ( tmp > 255 ) tmp = 255;
		d_color_buffer[tid].z = tmp;
	}
}



void BuildImage(uchar3* h_color_buffer, int r_w, int r_h)
{
	int num_rays = r_w*r_h;

	uchar3 *d_color_buffer;
	cudaMalloc(&d_color_buffer, r_w*r_h*sizeof(uchar3));


	InitEnv<<<1, 1>>>(r_w, r_h);
	RayTracing<<<num_rays/1024+1, 1024>>>(d_color_buffer);


	cudaMemcpy(h_color_buffer, d_color_buffer, r_w*r_h*sizeof(uchar3), cudaMemcpyDeviceToHost);
	cudaFree(d_color_buffer);
}