#pragma once
#include "math.h"

const int width = 1024;
const int height = 780;

const int windowX = 30;
const int windowY = 80;

const int image_size = width * height;

const int max_threads = 1024;

struct Material {
	float3 color;
	float reflect;
	float refract;
	float emit;
};

const Material basic_white = { {0.7f,0.7f,0.7f},0.0f,0.0f,0.0f };
const Material basic_orange = { { 0.7f,0.7f,0.7f },0.0f,0.0f,0.0f };
const Material basic_red = { { 0.9f,0.3f,0.2f },0.0f,0.0f,0.0f };
const Material basic_blue = { { 0.3f,0.4f,0.9f },0.0f,0.0f,0.0f };
const Material basic_green = { { 0.2f,0.9f,0.7f },0.0f,0.0f,0.0f };
const Material basic_yellow = { { 0.9f,0.85f,0.4f },0.0f,0.0f,0.0f };

const Material light_white = { { 1.33f,1.33f,1.33f },0.0f,0.0f,1.0f };
const Material light_red = { { 1.33f,0.22f,0.22f },0.0f,0.0f,0.8f };
const Material light_blue = { { 0.22f,0.22f,1.33f },0.0f,0.0f,0.8f };
const Material light_green = { { 0.22f,1.33f,0.22f },0.0f,0.0f,0.8f };

const Material mirror_ideal = { { 0.8f,0.8f,0.8f },1.0f,0.0f,0.0f };

const Material mirror_50 = { { 0.9f,0.3f,0.2f },0.5f,0.0f,0.0f };

const Material glass_ideal = { { 0.99f,0.99f,0.99f },1.0f,0.7f,0.0f };

struct Polygon {
	float3 a, b, c;
	Material material;
	bool is_plane;
	float3 norm;
};

__host__ __device__ float3 vec3(float x = 0, float y = 0, float z = 0);

void add_polygon(float3 a, float3 b, float3 c, Material material, bool is_plane = false);
void add_cube(float3 center, float3 sizes, Material material, float angle = 0.0f);
void build_room(float3 sizes);
void path_tracing(int iteration, float3 * buffer);