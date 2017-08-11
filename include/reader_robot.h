#pragma once

#include <opencv2/core/core.hpp>
#include "Eigen/Geometry"


class ReaderRobot {
public:

	ReaderRobot(std::string path);

	void read_pose (Eigen::Matrix<double, 4, 4, Eigen::RowMajor> &J4_pose,
		Eigen::Matrix<double, 4, 4, Eigen::RowMajor> &J5_pose,
		Eigen::Matrix<double, 4, 4, Eigen::RowMajor> &bHe_pose,
		float &jaw
		);

protected:

	std::vector<cv::Mat> bHe_poses;

	// Shaft
	std::vector<cv::Mat> J4_poses;

	// Logo
	std::vector<cv::Mat> J5_poses;

	// Radian
	std::vector<float> jaws;

	int frame_count;

	cv::FileStorage fs;

};

