import torch
from torch import Tensor
from pytorch3d import transforms as T

@torch.jit.script
def raydist3d_sphere(
    obst_pos: Tensor, # [n_envs, n_spheres, 3]
    obst_r: Tensor, # [n_envs, n_spheres]
    start: Tensor, # [n_envs, n_rays, 3]
    direction: Tensor, # [n_envs, n_rays, 3]
    max_dist: float
) -> Tensor:
    """Compute the ray distance based on the start of the ray,
    the direction of the ray, and the position and radius of 
    the sphere obstacles.

    Args:
        obst_pos (torch.Tensor): The center position of the sphere obstacles.
        obst_r (torch.Tensor): The radius of the sphere obstacles.
        start (torch.Tensor): The start point of the ray.
        direction (torch.Tensor): The direction of the ray.
        max_dist (float): The maximum traveling distance of the ray.

    Returns:
        torch.Tensor: The distance of the ray to the nearest obstacle's surface.
    """    
    rel_pos = obst_pos.unsqueeze(1) - start.unsqueeze(2) # [n_envs, n_rays, n_spheres, 3]
    rel_dist = torch.norm(rel_pos, dim=-1) # [n_envs, n_agents, n_spheres]
    costheta = torch.cosine_similarity(rel_pos, direction.unsqueeze(2), dim=-1) # [n_envs, n_rays, n_spheres]
    sintheta = torch.where(costheta>0, torch.sqrt(1 - costheta**2), 0.9) # [n_envs, n_rays, n_spheres]
    dist_center2ray = rel_dist * sintheta # [n_envs, n_rays, n_spheres]
    obst_r = obst_r.unsqueeze(1) # [n_envs, 1, n_spheres]
    raydist = rel_dist * costheta - torch.sqrt(torch.pow(obst_r, 2) - torch.pow(dist_center2ray, 2)) # [n_envs, n_rays, n_spheres]
    valid = torch.logical_and(dist_center2ray < obst_r, costheta > 0)
    valid = torch.logical_and(valid, raydist < max_dist)
    # valid = (dist_center2ray < obst_r) & (costheta > 0) & (raydist < max_dist) # [n_envs, n_rays, n_spheres]
    raydist_valid = torch.where(valid, raydist, max_dist) # [n_envs, n_rays, n_spheres]
    return raydist_valid.min(dim=-1).values # [n_envs, n_rays]

@torch.jit.script
def raydist3d_cube(
    box_min: Tensor, # [n_envs, n_cubes, 3]
    box_max: Tensor, # [n_envs, n_cubes, 3]
    start: Tensor, # [n_envs, n_rays, 3]
    direction: Tensor, # [n_envs, n_rays, 3]
    max_dist: float
) -> Tensor:
    """Compute the ray distance based on the start of the ray,
    the direction of the ray, and the position and radius of 
    the cubic obstacles.

    Args:
        obst_pos (torch.Tensor): The center position of the sphcubicere obstacles.
        obst_lwh (torch.Tensor): The length, width and height of the cubic obstacles.
        start (torch.Tensor): The start point of the ray.
        direction (torch.Tensor): The direction of the ray.
        max_dist (float): The maximum traveling distance of the ray.

    Returns:
        torch.Tensor: The distance of the ray to the nearest obstacle's surface.
    """
    _tmin = (box_min.unsqueeze(1) - start.unsqueeze(2)) / direction.unsqueeze(2) # [n_envs, n_rays, n_cubes, 3]
    _tmax = (box_max.unsqueeze(1) - start.unsqueeze(2)) / direction.unsqueeze(2) # [n_envs, n_rays, n_cubes, 3]
    tmin = torch.where(direction.unsqueeze(2) < 0, _tmax, _tmin) # [n_envs, n_rays, n_cubes, 3]
    tmax = torch.where(direction.unsqueeze(2) < 0, _tmin, _tmax) # [n_envs, n_rays, n_cubes, 3]
    tentry = torch.max(tmin, dim=-1).values # [n_envs, n_rays, n_cubes]
    texit = torch.min(tmax, dim=-1).values # [n_envs, n_rays, n_cubes]
    valid = torch.logical_and(tentry <= texit, texit >= 0) # [n_envs, n_rays, n_cubes]
    raydist = torch.where(valid, tentry, max_dist) # [n_envs, n_rays, n_cubes]
    return raydist.min(dim=-1).values # [n_envs, n_rays]

class Camera:
    def __init__(self, cfg, device):
        self.H = cfg.height
        self.W = cfg.width
        self.hfov = cfg.horizontal_fov
        self.vfov = self.hfov * self.H / self.W
        self.max_dist = cfg.max_dist
        self.device = device
        self.ray_directions = self._get_ray_directions_plane()
        # self.ray_directions = self._get_ray_directions_sphere()
    
    def get_raydist(
        self,
        sphere_pos: Tensor, # [n_envs, n_spheres, 3]
        sphere_r: Tensor, # [n_envs, n_spheres]
        box_min: Tensor, # [n_envs, n_cubes, 3]
        box_max: Tensor, # [n_envs, n_cubes, 3]
        start: Tensor, # [n_envs, n_rays, 3]
        quat_xyzw: Tensor, # [n_envs, 4]
    ) -> Tensor: # [n_envs, n_rays]
        ray_directions = self.world2body(quat_xyzw) # [n_envs, n_rays, 3]
        raydist_sphere: Tensor = raydist3d_sphere(sphere_pos, sphere_r, start, ray_directions, self.max_dist) # [n_envs, n_rays]
        raydist_cube: Tensor = raydist3d_cube(box_min, box_max, start, ray_directions, self.max_dist) # [n_envs, n_rays]
        raydist = torch.minimum(raydist_sphere, raydist_cube) # [n_envs, n_rays]
        depth = 1. - raydist.reshape(-1, self.H, self.W) / self.max_dist # [n_envs, H, W]
        return depth
    
    def world2body(self, quat_xyzw: Tensor) -> Tensor:
        quat_wxyz = quat_xyzw.roll(1, dims=-1) # [n_envs, 4]
        quat_wxyz = quat_wxyz.unsqueeze(1).expand(-1, self.H*self.W, -1) # [n_envs, n_rays, 4]
        return T.quaternion_apply(quat_wxyz, self.ray_directions.view(1, -1, 3)) # [n_envs, n_rays, 3]
        
    def _get_ray_directions_sphere(self):
        pitch = torch.linspace(0.5*self.vfov, -0.5*self.vfov, self.H, device=self.device) * torch.pi / 180
        yaw = torch.linspace(-0.5*self.hfov, 0.5*self.hfov, self.W, device=self.device) * torch.pi / 180
        pitch, yaw = torch.meshgrid(pitch, yaw)
        roll = torch.zeros_like(pitch)
        ypr = torch.stack([yaw, pitch, roll], dim=-1)
        rotmat = T.euler_angles_to_matrix(ypr, convention="ZYX") # [H, W, 3, 3]
        forward = Tensor([[[1., 0., 0.]]], device=self.device).expand(self.H, self.W, -1).unsqueeze(-1) # [H, W, 3, 1]
        directions = rotmat.transpose(-1, -2) @ forward # [H, W, 3, 1]
        return directions.squeeze(-1) # [H, W, 3]

    def _get_ray_directions_plane(self):
        import math
        forward = torch.tensor([[[1., 0., 0.]]], device=self.device).expand(self.H, self.W, -1) # [H, W, 3]
        
        vangle = 0.5 * self.vfov * torch.pi / 180
        vertical_offset = torch.linspace(math.tan(vangle), -math.tan(vangle), self.H, device=self.device).reshape(-1, 1, 1) # [H, 1, 1]
        zero = torch.zeros_like(vertical_offset)
        vertical_offset = torch.concat([zero, zero, vertical_offset], dim=-1) # [H, 1, 3]
        
        hangle = 0.5 * self.hfov * torch.pi / 180
        horizontal_offset = torch.linspace(math.tan(hangle), -math.tan(hangle), self.W, device=self.device).reshape(1, -1, 1) # [1, W, 1]
        zero = torch.zeros_like(horizontal_offset)
        horizontal_offset = torch.concat([zero, horizontal_offset, zero], dim=-1) # [1, W, 3]
        
        return forward + vertical_offset + horizontal_offset # [H, W, 3]
