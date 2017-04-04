#include "FlowExtractor.h"
#include <iostream>
#include "assert.h"

using namespace cv;
using namespace cv::gpu;

FlowExtractor::FlowExtractor()
{
    setDevice(0);
}


FlowExtractor::FlowExtractor(const std::vector<Size>& sizes)
{
    FlowExtractor();
    buildPyramidGpu(sizes);
}

void FlowExtractor::buildPyramidGpu(const std::vector<Size>& sizes)
{
    _nLayersPyramid = sizes.size();

    _gpu_pyr1.resize(_nLayersPyramid);
    _gpu_pyr2.resize(_nLayersPyramid);
    _gpu_pyr_flow_x.resize(_nLayersPyramid);
    _gpu_pyr_flow_y.resize(_nLayersPyramid);

    for(int i = 0; i < _nLayersPyramid; i++)
    {
        _gpu_pyr1[i].create(sizes[i], CV_8UC1);
        _gpu_pyr2[i].create(sizes[i], CV_8UC1);
        _gpu_pyr_flow_x[i].create(sizes[i], CV_32FC2);
        _gpu_pyr_flow_y[i].create(sizes[i], CV_32FC2);
    }
}

Mat FlowExtractor::computeFlow(const Mat& grey1, const Mat& grey2)
{
    // Upload to the GPU
    _gpu_grey1.upload(grey1);
    _gpu_grey2.upload(grey2);

    // Actual computation of the optical flow
    _flow_tvl1(_gpu_grey1, _gpu_grey2, _gpu_flow_x, _gpu_flow_y);

    // Download back to the CPU
    _gpu_flow_x.download(_flow_x);
    _gpu_flow_y.download(_flow_y);

    // Return a matrix with 2 channels, flow x and y direction
    Mat flow_merged = mergeFlowChannels(_flow_x, _flow_y);
    return flow_merged;
}

std::vector<Mat> FlowExtractor::computeFlowPyramid(const std::vector<Mat>& grey_pyr1, const std::vector<Mat>& grey_pyr2)
{
    assert(grey_pyr1.size() == grey_pyr2.size());

    // One by one process the layers in the pyramid
    std::vector<Mat> flow_pyr(_nLayersPyramid);
    for (int i = 0; i < _nLayersPyramid; i++)
    {
        assert(grey_pyr1[i].size() == grey_pyr2[i].size());
        flow_pyr[i] = computeFlow(grey_pyr1[i], grey_pyr2[i]);
    }

    return flow_pyr;
}

Mat FlowExtractor::mergeFlowChannels(const Mat& flow_x, const Mat& flow_y) const
{
    std::vector<Mat> flow_ch;
    flow_ch.push_back(flow_x);
    flow_ch.push_back(flow_y);
    Mat flow_merged;
    merge(flow_ch, flow_merged);
    return flow_merged;
}
