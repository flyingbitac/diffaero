from setuptools import setup, find_packages

setup(
    name='quaddif',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'torch>=2.0.0',
        'numpy',
        'gym',
        'matplotlib',
        'pytorch3d',
        'hydra-core',
        'line_profiler',
        'tqdm',
        'tensorboard==2.12.0',
        'opencv-python',
        'isaacgym',
        'welford_torch',
        'opencv-python'
        'moviepy',
        'imageio',
        'imageio-ffmpeg'
    ],
    author='Xinhong Zhang',
    author_email='zxh0916@126.com',
    description='',
    url=''
)
