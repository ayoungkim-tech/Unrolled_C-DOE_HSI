# MatConvNet: CNNs for MATLAB

**MatConvNet** is a MATLAB toolbox implementing *Convolutional Neural
Networks* (CNNs) for computer vision applications. It is simple,
efficient, and can run and learn state-of-the-art CNNs. Several
example CNNs are included to classify and encode images. Please visit
the [homepage](http://www.vlfeat.org/matconvnet) to know more.

## Requirements (assuming Windows OS and Matlab 2022b)
* [Microsoft Visual C++ 2019](https://my.visualstudio.com/Downloads?q=Visual%20Studio%202019): The compatible version can be found at [here](https://www.mathworks.com/content/dam/mathworks/mathworks-dot-com/support/sysreq/files/system-requirements-release-2022b-supported-compilers.pdf)
* [CUDA Toolkit 11.0 ](https://developer.nvidia.com/cuda-10.1-download-archive-base?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exelocal): The compatible version can be found at [here](https://docs.nvidia.com/cuda/archive/10.1/cuda-installation-guide-microsoft-windows/index.html)

## Set up
Add the following paths of cl.exe at Windows settings/system/About/Advanced system settings/ Environment Variables / System Variables /Path.
| Variable  | Value  |
| :--------: | :----------- | 
| VS_PATH | C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\VC\Tools\MSVC\14.29.30133\bin\Hostx86\x86 |
| CUDA_PATH | C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0 |
| MW_NVCC_PATH | C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\bin\nvcc |

## Compile
1. Edit following lines at matconvnet-1.0-beta25_gpu -10.0\matlab\vl_compilenn.m,
* opts.enableImreadJpeg = false; <- line 172
* % flags.base{end+1} = '-O' ; <- comment line 340 for GPU ver.
* flags.mexlink = {'-lmwblas'}; <- line 359
* flags.base, flags.mexlink, ... <- line 621

2. Compile at Matlab Command Window
```
cd('matconvnet-1.0-beta25_gpu -10.0')
run('matlab\vl_setupnn.m');
mex -setup C++ % pick Microsoft Visual C++ 2019
addpath matlab
%% Choose CPU or GPU ver. GPU ver. is needed for training.
vl_compilenn % CPU ver.
vl_compilenn('enableGpu', true) % GPU ver.
```
