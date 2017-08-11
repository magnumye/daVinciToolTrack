#pragma once


#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "stereo_base.h"


class ReaderCamera {
public:

	ReaderCamera(std::string path, std::string stereo_calib_path);

	~ReaderCamera();

	void read_curr_image(cv::Mat &img_out);


protected:

	cv::Size img_sz;

	cv::Mat curr_image;

	std::vector<cv::Mat> video_data_L;

	std::vector<cv::Mat> video_data_R;

	int frame_count;

	int total_num_frame;

	// Stereo class
	StereoBase *m_stereo_;

	cv::VideoCapture vid_cap;

};

