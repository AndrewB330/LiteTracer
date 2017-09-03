#include "litetracer.h"
#include "device_launch_parameters.h"
#include "curand.h"
#include "curand_kernel.h"
#include <algorithm>
using std::swap;
#define RANDOM_PIXELS

int polygons_number = 0;
int polygons_allocated = 0;
Polygon * polygons_d;
float3 * buffer_d;
float3* accumulator_d;

float3 camera = { 2,4,5.5f };
float3 direction = normalize(make_float3(0, 1.8f, 0) - camera);

__constant__ float3 world_up = { 0,1,0 };
__constant__ float cam_to_screen = 0.9f;//~fov
__constant__ float scale = 1.0f / (width < height ? width : height);
__constant__ float EPS = 1e-6f;
__constant__ float INF = 1e6f;

__constant__ float magic_factor = 1.4f;//magic factor for light intensity tuning

__constant__ int max_bounces = 18;

void reallocate()
{
	Polygon * polygons = new Polygon[polygons_number];
	cudaMemcpy(polygons, polygons_d, polygons_number * sizeof(Polygon), cudaMemcpyDeviceToHost);
	cudaFree(polygons_d);
	cudaMalloc(&polygons_d, polygons_allocated * sizeof(Polygon));
	cudaMemcpy(polygons_d, polygons, polygons_number * sizeof(Polygon), cudaMemcpyHostToDevice);
	delete polygons;
}

__host__ __device__ float3 vec3(float x, float y, float z)
{
	return make_float3(x, y, z);
}

void add_polygon(float3 a, float3 b, float3 c, Material material, bool is_plane)
{
	if (polygons_number >= polygons_allocated)
	{
		polygons_allocated++;
		polygons_allocated *= 1.3;//in advance, std::vector-style
		reallocate();
	}
	Polygon polygon = { a,b,c,material,is_plane, normalize(cross(b - a,c - a)) };
	cudaMemcpy(polygons_d + polygons_number++, &polygon, sizeof(Polygon), cudaMemcpyHostToDevice);
}

void add_cube(float3 center, float3 sizes, Material material, float angle)
{
	float3 verticies[8];
	for (int i = 0; i < 8; i++)
	{
		verticies[i].x = (2 * (bool)(i & 1) - 1)*sizes.x * 0.5f;
		verticies[i].y = (2 * (bool)(i & 2) - 1)*sizes.y * 0.5f;
		verticies[i].z = (2 * (bool)(i & 4) - 1)*sizes.z * 0.5f;
		verticies[i] = rotate_y(verticies[i], angle);
		verticies[i] += center;
	}
	//verts in reversed order
	int3 polygon_verts[12] = {
		{0,1,3},//front
		{0,3,2},//front
		{1,5,7},//right
		{1,7,3},//right
		{4,0,2},//left
		{4,2,6},//left
		{2,3,7},//top
		{2,7,6},//top
		{4,5,1},//bottom
		{4,1,0},//bottom
		{5,4,6},//back
		{5,6,7}//back
	};
	for (int i = 0; i < 12; i++)
	{
		float3 a = verticies[polygon_verts[i].z];
		float3 b = verticies[polygon_verts[i].y];
		float3 c = verticies[polygon_verts[i].x];
		add_polygon(a, b, c, material, false);
	}
}

void build_room(float3 sizes)
{
	float3 verticies[8];
	for (int i = 0; i < 8; i++)
	{
		verticies[i].x = (2 * (bool)(i & 1) - 1)*sizes.x  * 0.5f;
		verticies[i].y = (2 * (bool)(i & 2) - 0)*sizes.y  * 0.5f;
		verticies[i].z = (2 * (bool)(i & 4) - 1)*sizes.z  * 0.5f;
	}
	int3 wall_verts[6] = {
		{ 0,1,3 },//front
		{ 1,5,7 },//right
		{ 4,0,2 },//left
		{ 2,3,7 },//top
		{ 4,5,1 },//bottom
		{ 5,4,6 },//back
	};
	Material materials[6] = {
		mirror_ideal,
		basic_yellow,
		basic_blue,
		light_white,
		basic_white,
		basic_white,
	};
	for (int i = 0; i < 6; i++)
	{
		float3 a = verticies[wall_verts[i].x];
		float3 b = verticies[wall_verts[i].y];
		float3 c = verticies[wall_verts[i].z];
		add_polygon(a, b, c, materials[i], 1);
	}
}

inline __host__ __device__ float intersect(Polygon poly, float3 origin, float3 direction)
{
	//returns distance from origin to intersection point on poligon
	float distance = -dot((origin - poly.a), poly.norm) / dot(direction, poly.norm);
	if (distance < EPS || (dot(poly.norm, direction) > EPS && poly.is_plane))
		return -INF;
	if (poly.is_plane)
		return distance;
	float3 hit = origin + direction*distance;
	if (dot(cross(poly.b - poly.a, hit - poly.a), poly.norm) > EPS &&
		dot(cross(poly.c - poly.b, hit - poly.b), poly.norm) > EPS &&
		dot(cross(poly.a - poly.c, hit - poly.c), poly.norm) > EPS)
		return distance;
	return -INF;
}

inline __device__ float3 randomCosin(float3 n, curandState & state)
{
	//cosine weighted random direction
	float r1 = 2 * PI * curand_uniform(&state);
	float r2 = curand_uniform(&state);
	float r2s = sqrtf(r2);
	float3 u = normalize(cross((fabs(n.x) > .1 ? make_float3(0, 1, 0) : make_float3(1, 0, 0)), n));
	float3 v = cross(n, u);
	return normalize(u*cos(r1)*r2s + v*sin(r1)*r2s + n*sqrtf(1 - r2));
}

__global__ void render_kernel(int iteration, int seed, int polygons_number, Polygon * polygons, float3 * accumulator, float3 * output, float3 camera, float3 cam_direction)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= image_size) return;
	curandState state;
	curand_init(seed * idx + iteration * idx + iteration + seed, 0, 0, &state);
#ifdef RANDOM_PIXELS
	//Anti-aliasing effect
	float x = scale*(idx % width - width / 2 + curand_uniform(&state));
	float y = scale*(idx / width - height / 2 + curand_uniform(&state));
#else
	float x = scale*(idx % width - width / 2);
	float y = scale*(idx / width - height / 2);
#endif
	float3 right = cross(cam_direction, world_up);
	float3 up = cross(right, cam_direction);
	float3 ray_origin = camera;
	float3 ray_direction = normalize(cam_to_screen*cam_direction + x*right + y*up);

	float3 mask = { 1.0f,1.0f,1.0f };//color mask
	float3 accumulated = { 0,0,0 };//accumulate color while a ray bounces
	for (int bounce = 0; bounce < max_bounces; bounce++)
	{
		float nearest = INF;
		int nearest_id = -1;
		for (int i = 0; i < polygons_number; i++)
		{
			float distance = intersect(polygons[i], ray_origin, ray_direction);
			if (distance > EPS && distance < nearest)
			{
				nearest = distance;
				nearest_id = i;
			}
		}
		if (nearest_id == -1) break;
		float3 n = polygons[nearest_id].norm;
		ray_origin += ray_direction*nearest;
		float3 random_direction = randomCosin(n, state);
		float3 reflected_direction = reflect(ray_direction, n);
		float reflection_factor = polygons[nearest_id].material.reflect;


		float refraction_index = polygons[nearest_id].material.refract;
		//if refractive material
		if (refraction_index > EPS)
		{
			//if ray goes from inside
			if (dot(n, ray_direction) > EPS)
				refraction_index = 1.0f / refraction_index;
			else
				n *= -1;
			float nsin = length(cross(n, ray_direction)) * refraction_index;//sin of outcoming ray
			if (fabsf(nsin) < 1.0f)
			{
				float ntg = nsin / sqrtf(1.0f - nsin*nsin);//tan of outcoming ray
				ray_direction = normalize(n + ntg*normalize(cross(cross(n, ray_direction), n)));
				ray_origin += n*EPS;
			}
			else
			{
				//total internal reflection
				ray_direction = reflected_direction;
				ray_origin -= n*EPS;
			}
		}
		else
		{
			//calculating of outcoming ray using reflection_factor
			ray_direction = normalize((1.0f - reflection_factor)*random_direction + reflection_factor*reflected_direction);
			ray_origin += n*EPS;
		}

		accumulated += mask*polygons[nearest_id].material.emit * polygons[nearest_id].material.color;

		mask *= polygons[nearest_id].material.color;
		mask *= dot(n, ray_direction)*(1.0 - reflection_factor) + reflection_factor;
		mask *= magic_factor*(1.0 - reflection_factor) + reflection_factor;
	}
	accumulator[idx] += accumulated;
	output[idx] = accumulator[idx] / (iteration + 1);
}

void path_tracing(int iteration, float3 * buffer)
{
	if (iteration == 0 || accumulator_d == 0)
	{
		cudaFree(accumulator_d);
		cudaFree(buffer_d);
		cudaMalloc(&accumulator_d, image_size * sizeof(float3));
		cudaMemset(accumulator_d, 0, image_size * sizeof(float3));
		cudaMalloc(&buffer_d, image_size * sizeof(float3));
	}
	int threads = max_threads;
	int blocks = (image_size + threads - 1) / threads;
	render_kernel << <blocks, threads >> > (
		iteration, //iteration
		rand() * RAND_MAX +  rand(), //seed
		polygons_number, //number of polygons
		polygons_d, //polygons array
		accumulator_d, //pixels accumulator
		buffer_d, //output pixel buffer
		camera, //camera position
		direction //direction from camera to target
		);
	cudaMemcpy(buffer, buffer_d, image_size * sizeof(float3), cudaMemcpyDeviceToHost);
}
