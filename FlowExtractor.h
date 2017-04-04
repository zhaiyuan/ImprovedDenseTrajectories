#pragma once

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/gpu/gpu.hpp"

using namespace cv;
using namespace cv::gpu;

class FlowExtractor
{

public:
    FlowExtractor();
    FlowExtractor(const std::vector<Size>& sizes);
    void buildPyramidGpu(const std::vector<Size>& sizes);
    Mat computeFlow(const Mat& grey1, const Mat& grey2);
    std::vector<Mat> computeFlowPyramid(const std::vector<Mat>& grey_pyr1, const std::vector<Mat>& grey_pyr2);
    Mat mergeFlowChannels(const Mat& flow_x, const Mat& flow_y) const;

private:

    // Number of layers in the pyramid
    int _nLayersPyramid;

    // CPU matrices
    Mat _flow_x, _flow_y;
    std::vector<Mat> _grey_pyr1, _grey_pyr2;

    // GPU matrices
    GpuMat _gpu_grey1, _gpu_grey2;
    GpuMat _gpu_flow_x, _gpu_flow_y;

    std::vector<GpuMat> _gpu_pyr1, _gpu_pyr2;
    std::vector<GpuMat> _gpu_pyr_flow_x, _gpu_pyr_flow_y;

    // Flow Extractor
    OpticalFlowDual_TVL1_GPU _flow_tvl1;

};
