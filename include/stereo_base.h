#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class StereoBase
{
public:
	StereoBase ();
	StereoBase (const std::string calib_filename);
// 	virtual ~StereoBase();

	// Set input left and right image
    virtual void setInputImages(const cv::Mat &leftImage, const cv::Mat &rightImage);
	void setScale(unsigned s);

    virtual void calcPointCloud(cv::Mat &cld) {}

	// Undistort and triangulate image points to 3D point
	// Input: points1 and points2 are 1xN array contains left and right image points correspondingly
	// Output: point3d is output 3D points
	void mTriangulatePoints(
		cv::InputArray points1, cv::InputArray points2, cv::OutputArray point3d);


	// Get camera parameters
	inline cv::Mat& getR1() { return R1; }
	inline cv::Mat& getR2() { return R2; }
	inline cv::Mat& getP1() { return P1; }
	inline cv::Mat& getP2() { return P2; }
	inline cv::Mat& get_cloud() { return cloud; }
	inline cv::Mat& getcameraMatrix(const int lr = 0) { return cameraMatrix[lr]; }
	inline cv::Mat& getdistCoeffs(const int lr = 0) { return distCoeffs[lr]; }

	cv::Mat rectifyL(const cv::Mat &input);
	cv::Mat rectifyR(const cv::Mat &input);

	void getRectifiedImage(cv::Mat &iLeftImg, cv::Mat &iRightImg);
	void getDownRectifiedImage(cv::Mat &oLeftImg, cv::Mat &oRightImg);

	// SAVE
	static void writecvPoint3fToPLYFile (const std::vector<cv::Point3f> &points, 
		const std::string &filename);

	// save point cloud to ply format file
	// \param cloud the 3-channel Mat save point cloud
	// \param normal the normal for the cloud, cloud.size()=normal.size()
	// \param color the color image, cloud.size()=color.size()
	// \param filename
	static void writePointCloudToPLYFile (const cv::Mat &cloud, 
		const cv::Mat &normals, 
		const cv::Mat &color,
		const std::string &filename);


protected:
	// Image size
	cv::Size image_size;
	// Camera parameters
	cv::Mat cameraMatrix[2];
	cv::Mat distCoeffs[2];
	cv::Mat R, T, R1, R2, P1, P2, E, F, Q;

	// Raw left and right image
	cv::Mat leftImage, rightImage;
	// undistorted image
	cv::Mat recLeftImage, recRightImage;
	// rectification map
	cv::Mat rmap[2][2];
	// Raw Disparity Image
	cv::Mat rawDisparity;

	// Left disparity map (float)
	cv::Mat disp_left;
	// Right disparity map (float)
	cv::Mat disp_right;

	// 3D point cloud
	cv::Mat cloud;

	/************************************************************************/
	/* For downsampling                                                     */
	/************************************************************************/
	// Downsample image
	cv::Mat dwnleftImage, dwnrightImage;
	// undistorted image
	cv::Mat dwnrecLeftImage, dwnrecRightImage;
	// Color map
	cv::Mat dwnleftColor, dwnrightColor;
	// Scale factor for downsampling, 2 means half the image
	unsigned scale_dwn;
	// image size after downsample
	cv::Size2i image_size_dwn;
	// Camera matrix for downsampling
	cv::Mat cameraMatrix_dwn[2];
	// rectification map
	cv::Mat dwnrmap[2][2];
	cv::Mat dwndistCoeffs[2];
	cv::Mat dwnR, dwnT, dwnR1, dwnR2, dwnP1, dwnP2, dwnE, dwnF, dwnQ;

private:
};
