from setuptools import setup, find_packages

setup(
    name='quaddif',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'torch>=2.0.0',
        'tensordict',
        'taichi',
        'numpy',
        'gym',
        'matplotlib',
        'pytorch3d',
        'hydra-core',
        'line_profiler',
        'tqdm',
        'tensorboard',
        'tensorboardX',
        'wandb',
        'opencv-python',
        'welford_torch',
        'opencv-python',
        'moviepy',
        'imageio',
        'imageio-ffmpeg',
        'onnx',
        'ncnn'
    ],
    author='Xinhong Zhang',
    author_email='zxh0916@126.com',
    description='',
    url=''
)
