#include <iostream>
#include <fstream>
#include "GL\freeglut.h"
#include "cuda.h"
#include "math.h"
#include "lite_tracer.h"
#include "gpu_timer.h"

float buffer[image_size * 3];

void render()
{
	glClear(GL_COLOR_BUFFER_BIT);
	glClearColor(0.05f, 0.0f, 0.06f, 1.0f);
	glDrawPixels(width, height, GL_RGB, GL_FLOAT, buffer);
	glFlush(); glutSwapBuffers();
}

void scene1()
{
	add_cube(vec3(+1.6f, 0.9f, +0.6f), { 0.8f,0.8f,0.8f }, light_green, 0.3);
	add_cube(vec3(+0.0f, 0.9f, +0.1f), { 1.1f,1.1f,1.1f }, basic_red, 0.7);
	add_cube(vec3(-1.8f, 0.9f, +0.3f), { 0.8f,0.8f,0.8f }, light_red, 0.9);
	add_cube(vec3(-0.65f, 1.9f, +1.9f), { 1.3f,1.1f,0.1f }, glass_ideal, 0.2);
}

void scene2()
{
	add_cube(vec3(-0.6, 0.9f, +0.3f), { 3.5f,0.8f,0.8f }, basic_blue, 0.0);
	add_cube(vec3(+1.0, 1.1f, -0.8f), { 2.5f,0.8f,0.8f }, mirror_50, 0.0);
	add_cube(vec3(-0.6f, 1.3f, 1.2f), { 2.1f,1.8f,0.2f }, glass_ideal, 0.0);
	add_cube(vec3(+0.1f, 1.6f, -0.2f), { 1.8f,1.8f,0.2f }, glass_ideal, 0.0);
}

int main(int argc, char** argv)
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
	glutInitWindowSize(width, height);
	glutInitWindowPosition(windowX, windowY);
	glutCreateWindow("LiteGL3000");
	glutDisplayFunc(render);
	glLoadIdentity();
	glOrtho(0, width, 0, height, 0, 1);


	build_room(vec3(6, 5, 6));
	scene2();

	float accumulated_time = 0.0f;
	for (int iter = 0;; iter++)
	{
		GpuTimer timer;
		timer.Start();
		path_tracing(iter, (float3*)buffer);
		if ((iter + 1) % 30 == 0)
		{
			printf("Time elapsed = %g ms\n", accumulated_time);
			printf("Average FPS = %g fps\n", iter / accumulated_time * 1000.f);
		}
		glutMainLoopEvent();
		glutPostRedisplay();
		timer.Stop();
		accumulated_time += timer.Elapsed();
	}
	return 0;
}