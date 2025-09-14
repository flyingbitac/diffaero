from setuptools import setup, find_packages

setup(
    name='diffaero',
    version='0.1',
    # packages=find_packages(),
    packages=[".", "env", "algo", "network", "utils", "script"],
    install_requires=[
        'torch>=2.0.0',
        'tensordict',
        'taichi',
        'tqdm',
        'hydra-core',
        'hydra-joblib-launcher',
        'hydra_colorlog',
        'hydra-optuna-sweeper',
        'welford_torch',
        'line_profiler',
        'tensorboard',
        'tensorboardX',
        'torch-tb-profiler',
        'wandb',
        'gpustat',
        'opencv-python',
        'pytorch3d@git+https://github.com/facebookresearch/pytorch3d.git@stable#egg=pytorch3d',
        'open3d',
        'numpy',
        'moviepy==1.0.3',
        'imageio',
        'imageio-ffmpeg',
        'matplotlib',
        'onnx',
        'onnxruntime'
    ],
    author='Xinhong Zhang',
    author_email='xhzhang@bit.edu.cn',
    description='',
    url='https://github.com/flyingbitac/diffaero'
)
