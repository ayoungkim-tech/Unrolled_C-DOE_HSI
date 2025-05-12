# Snapshot Hyperspectral Imaging with Co-designed Optics, Color Filter Array, and Unrolled Network
This repository contains the scripts associated with the paper "Snapshot Hyperspectral Imaging with Co-designed Optics, Color Filter Array, and Unrolled Network".

If you find our work useful in your research, please cite:

```
CITATION BIB
```

## Abstract
We propose a novel snapshot hyperspectral imaging method that incorporates co-designed optics, a color filter array (CFA), and an unrolled post-processing network through end-to-end learning. The camera optics consist of a fixed refractive lens and a diffractive optical element (DOE). The learned DOE and CFA efficiently encode the hyperspectral data cube on the sensor via phase and amplitude modulation at the camera aperture and sensor planes, respectively. Subsequently, the unrolled network reconstructs the hyperspectral images from the sensor signal with high accuracy. We conduct extensive simulations to analyze and validate the performance of the proposed method for several CFA models and in non-ideal imaging conditions. We demonstrate that the Gaussian model is effective for parameterizing the spectral transmission functions of CFA pixels, providing high reconstruction accuracy and being relatively easy to implement. Furthermore, we show that learned CFA patterns are effective when optimally coupled with co-designed diffractive-refractive optics. We evaluate the robustness of our method against sensor noise and potential inaccuracies in the fabrication of the DOE and CFA. Our results show that our method achieves superior reconstruction quality compared to state-of-the-art methods, excelling in both spatial and spectral detail recovery, and maintaining robustness against realistic noise levels. The code is available at the GitHub repository: https://github.com/ayoungkim-tech/Unrolled_C-DOE_HSI.git.

## License Information
LICENSE INFO

# Requirements

- MATLAB Version 7.3 or later
- [matconvnet-1.0-beta25_gpu -10.0](https://www.vlfeat.org/matconvnet/install/)
- [graphviz-2.38](https://graphviz.org/download/)
- [toolbox-master](https://pdollar.github.io/toolbox/)
- [HSI2RGB-master](https://github.com/JakobSig/HSI2RGB/tree/master)

## How to Execute the Code
1. Download requirements in the corresponding folders
2. Compile MatConvNet
3. Run net_test.m for testing, and net_train.m for training the network.

## Data

This work uses the following dataset. Please download the datasets and store them correctly in the corresponding dataset folder (TestData/TrainData).
- [ICVL Dataset](https://icvl.cs.bgu.ac.il/hyperspectral/) at TestData
- [KAIST Dataset](https://vclab.kaist.ac.kr/siggraphasia2017p1/index.html) at TrainData

## Structure of directories at main folder

| Directory  | Description  |
| :--------: | :----------- | 
| `Layers` | Functions forming the unrolled network structure | 
| `Util`    | Utility functions |
| `TestData`    |ICVL test data |
| `TrainData`    | KAIST train data |
| `Net`    | The trained network | 
| `Result`    |The test results after net_test.m|

## Contact
If you have any questions, please contact

* Ayoung Kim, ayoung.kim@tuni.fi
* Ugur Akpinar, ugur.akpinar@tuni.fi
* Erdem Sahin, erdem.sahin@tuni.fi
* Atanas Gotchev, atanas.gotchev@tuni.fi
