from isaacgym import gymapi

def get_sim_params(sim_cfg):
    sim_params = gymapi.SimParams()
    sim_params.substeps = sim_cfg.substeps
    up_axises = {0: gymapi.UP_AXIS_Y, 1: gymapi.UP_AXIS_Z}
    sim_params.up_axis = up_axises[sim_cfg.up_axis]
    sim_params.gravity = gymapi.Vec3(
        sim_cfg.gravity[0],
        sim_cfg.gravity[1],
        sim_cfg.gravity[2]
    )
    sim_params.use_gpu_pipeline = sim_cfg.use_gpu_pipeline
    
    exclude_keys = []
    
    physx_param = sim_cfg.physx
    for k, v in dict(physx_param).items():
        if hasattr(sim_params.physx, k) and v is not None:
            setattr(sim_params.physx, k, v)
        elif k not in exclude_keys:
            print(f'\033[31mWarning: {k} is not a valid physx param.\033[0m')
    return sim_params

def get_asset_options(asset_cfg):
    asset_options = gymapi.AssetOptions()
    exclude_keys = [
        'file', 'name', 'base_link_name', 'foot_name', 'penalize_contacts_on',
        'terminate_after_contacts_on', 'collision_mask', 'assets_per_env', 'segmentation_id',
        'walls', 'ground_plane']
    for k, v in dict(asset_cfg).items():
        if hasattr(asset_options, k) and v is not None:
            setattr(asset_options, k, v)
        elif k not in exclude_keys:
            print(f'\033[31mWarning: {k} is not a valid asset param.\033[0m')
    return asset_options

def get_camera_properties(camera_cfg):
    # Set Camera Properties
    camera_props = gymapi.CameraProperties()
    exclude_keys = ['enable', 'transform']
    for k, v in dict(camera_cfg).items():
        if hasattr(camera_props, k) and v is not None:
            setattr(camera_props, k, v)
        elif k not in exclude_keys:
            print(f'\033[31mWarning: {k} is not a valid asset param.\033[0m')
    # local camera transform
    local_transform = gymapi.Transform()
    # position of the camera relative to the body
    local_transform.p = gymapi.Vec3(*camera_cfg.transform.p)
    # orientation of the camera relative to the body
    local_transform.r = gymapi.Quat(*camera_cfg.transform.r)
    return camera_props, local_transform