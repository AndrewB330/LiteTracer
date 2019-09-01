# LiteTracer
Path tracing algorithm written on CUDA, uses freeglut library for displaying

## Features
* Diffuse surfaces
* Specular reflective surfaces
* Refractive surfaces
* Flexible materials system
* Anti-aliasing

### Renders:
![Demo](/Renders/glass.png?raw=true)

### Next render was made on previous version that has other path-tracing algorithm and supported sphere shape
![Demo](/Renders/mirrors.jpg?raw=true)

### Next renders I made on my new version of LiteTracer that support dynamic BVH (Bounding volume hierarchy, kd-tree). And can render up to 10,000,000 sphere and polygons on 24 FPS (on 1050Ti)
![Demo](/Renders/dragon3.png?raw=true)
![Demo](/Renders/dragon.png?raw=true)