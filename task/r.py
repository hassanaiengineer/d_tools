from enum import IntEnum

import taichi as ti
import taichi.math as tm

from .scene_data import SceneData
from .camera import Camera
from .ray import Ray, HitData
from .sampler import UniformSampler, BRDF
from .materials import Material


@ti.data_oriented
class A2Renderer:

    # Enumerate the different sampling modes
    class SampleMode(IntEnum):
        UNIFORM = 1
        BRDF = 2
        MICROFACET = 3

    def __init__( 
        self, 
        width: int, 
        height: int, 
        scene_data: SceneData
        ) -> None:

        self.RAY_OFFSET = 1e-6

        self.width = width
        self.height = height
        self.camera = Camera(width=width, height=height)
        self.canvas = ti.Vector.field(n=3, dtype=float, shape=(width, height))
        self.iter_counter = ti.field(dtype=float, shape=())
        self.scene_data = scene_data

        self.sample_mode = ti.field(shape=(), dtype=int)
        self.set_sample_uniform()


    def set_sample_uniform(self):    self.sample_mode[None] = self.SampleMode.UNIFORM
    def set_sample_brdf(self):       self.sample_mode[None] = self.SampleMode.BRDF
    def set_sample_microfacet(self): self.sample_mode[None] = self.SampleMode.MICROFACET


    @ti.kernel
    def render(self):
        '''
        - call generate_ray with jitter = True
        - progressively accumulate the pixel values in each canvas [x, y] position
        '''
        self.iter_counter[None] += 1
        for x, y in ti.ndrange(self.width, self.height):
            primary_ray = self.camera.generate_ray(x, y, jitter=True)
            color = self.shade_ray(primary_ray)
            self.canvas[x, y] += (color - self.canvas[x, y]) / self.iter_counter[None]

    def reset(self):
        self.canvas.fill(0.)
        self.iter_counter.fill(0.)


    @ti.func
    def shade_ray(self, ray: Ray) -> tm.vec3:
        color = tm.vec3(0.)

        '''
        You can change the structure of the shade ray function however you want as there will be computations that are the same for all 3 methods
        You can have your branching logic anywhere in the code
        '''

        # TODO: Implement Uniform Sampling
        if self.sample_mode[None] == int(self.SampleMode.UNIFORM):
            pass 
        
        # TODO: Implement BRDF Sampling
        elif self.sample_mode[None] == int(self.SampleMode.BRDF):
            pass     

        return color