#include "stereo_base.h"

StereoBase::StereoBase(const std::string calib_filename) :
	scale_dwn(1)
{
	std::cout << "Loading camera parameters..." << std::endl;
	// Read camera parameters
	cv::FileStorage fs;
	if (!fs.open(calib_filename, cv::FileStorage::READ))
	{
		std::cerr << "cannot open: " << calib_filename << std::endl;
		exit(1);
	}
	// Extrinsic
	fs["R"] >> R; fs["T"] >> T;
	fs["E"] >> E; fs["F"] >> F;
	// Intrinsic
	fs["M1"] >> cameraMatrix[0]; fs["D1"] >> distCoeffs[0];
	fs["M2"] >> cameraMatrix[1]; fs["D2"] >> distCoeffs[1];
	fs["image_Width"] >> image_size.width;
	fs["image_Height"] >> image_size.height;

	// Rectification parameters
	cv::stereoRectify(cameraMatrix[0], distCoeffs[0], cameraMatrix[1], distCoeffs[1],
		image_size, R, T, R1, R2, P1, P2, Q, 0, -1, image_size);
    Q.convertTo(Q, CV_32F);
	initUndistortRectifyMap(cameraMatrix[0], distCoeffs[0], R1, P1, image_size, CV_32FC1, rmap[0][0], rmap[0][1]);
	initUndistortRectifyMap(cameraMatrix[1], distCoeffs[1], R2, P2, image_size, CV_32FC1, rmap[1][0], rmap[1][1]);

	disp_left.create(image_size, CV_32FC1);
	disp_right.create(image_size, CV_32FC1);

	fs.release();
}

void StereoBase::setInputImages(const cv::Mat &input_left, const cv::Mat &input_right)
{
    input_left.copyTo(leftImage);
    input_right.copyTo(rightImage);
	if (scale_dwn != 1)
	{
        cv::resize(leftImage, dwnleftImage, image_size_dwn, 0, 0, CV_INTER_LINEAR);
        cv::resize(rightImage, dwnrightImage, image_size_dwn, 0, 0, CV_INTER_LINEAR);
	}
}


void StereoBase::setScale(unsigned s)
{
	if (s != scale_dwn && s != 1)
	{
		scale_dwn = s;
		image_size_dwn.width = image_size.width/scale_dwn;
		image_size_dwn.height = image_size.height/scale_dwn;

		cv::Mat pre_mat = cv::Mat::eye(3,3,CV_64FC1);
		pre_mat.at<double>(0,0) = 1/(double)scale_dwn, pre_mat.at<double>(1,1) = 1/(double)scale_dwn;
		cameraMatrix_dwn[0] = pre_mat * cameraMatrix[0];
		cameraMatrix_dwn[1] = pre_mat * cameraMatrix[1];
		dwndistCoeffs[0] = distCoeffs[0]; dwndistCoeffs[1] = distCoeffs[1];
		cv::stereoRectify(cameraMatrix_dwn[0], dwndistCoeffs[0], cameraMatrix_dwn[1], dwndistCoeffs[1],
			image_size_dwn, R, T, dwnR1, dwnR2, dwnP1, dwnP2, dwnQ, 0, -1, image_size_dwn);
        dwnQ.convertTo(dwnQ, CV_32F);
		initUndistortRectifyMap(cameraMatrix_dwn[0], dwndistCoeffs[0], dwnR1, dwnP1, image_size_dwn, CV_32FC1, dwnrmap[0][0], dwnrmap[0][1]);
		initUndistortRectifyMap(cameraMatrix_dwn[1], dwndistCoeffs[1], dwnR2, dwnP2, image_size_dwn, CV_32FC1, dwnrmap[1][0], dwnrmap[1][1]);
		disp_left.create(image_size_dwn, CV_32FC1);
		disp_right.create(image_size_dwn, CV_32FC1);
	}
}

void StereoBase::mTriangulatePoints(
	cv::InputArray points1, cv::InputArray points2, cv::OutputArray point3d)
{
	// undistort points
	cv::Mat points1_in = points1.getMat(), points2_in = points2.getMat();

	if (points1.channels() != 2)
	{
		std::cout << "Stereo::undistortTriangulatePoints, input not 2-channel array" << std::endl;
	}


	// triangulate
	cv::Mat point4d;
	cv::triangulatePoints(P1, P2, points1_in, points2_in, point4d);
	// convert from 4d to 3d
	// points1 and points2 have same type of point4d
	if (point4d.type() == CV_64F)
		point3d.create(point4d.cols, 1, CV_64FC3);
	else
		point3d.create(point4d.cols, 1, CV_32FC3);

	cv::Mat point3d_out = point3d.getMat();
	for (unsigned int i = 0; i < point4d.cols; i++)
	{
		if (point3d.type() == CV_64FC3)
		{
			point3d_out.at<cv::Vec3d>(i)[0] = point4d.at<double>(0,i) / point4d.at<double>(3,i);
			point3d_out.at<cv::Vec3d>(i)[1] = point4d.at<double>(1,i) / point4d.at<double>(3,i);
			point3d_out.at<cv::Vec3d>(i)[2] = point4d.at<double>(2,i) / point4d.at<double>(3,i);
		}
		else
		{
			point3d_out.at<cv::Vec3f>(i)[0] = point4d.at<float>(0,i) / point4d.at<float>(3,i);
			point3d_out.at<cv::Vec3f>(i)[1] = point4d.at<float>(1,i) / point4d.at<float>(3,i);
			point3d_out.at<cv::Vec3f>(i)[2] = point4d.at<float>(2,i) / point4d.at<float>(3,i);
		}
	}
}

cv::Mat StereoBase::rectifyL(const cv::Mat &input)
{
	cv::Mat output;
	if (input.cols == image_size.width)
	{
		cv::remap(input, output, rmap[0][0], rmap[0][1], cv::INTER_LINEAR);
	}
	else if (input.cols == image_size_dwn.width)
	{
		cv::remap(input, output, dwnrmap[0][0], dwnrmap[0][1], cv::INTER_LINEAR);
	}
	else
	{
		if (scale_dwn != 1)
			cv::remap(input, output, dwnrmap[0][0], dwnrmap[0][1], cv::INTER_LINEAR);
		else
			cv::remap(input, output, rmap[0][0], rmap[0][1], cv::INTER_LINEAR);
	}

	return output;
}

cv::Mat StereoBase::rectifyR(const cv::Mat &input)
{
	cv::Mat output;
	if (input.cols == image_size.width)
	{
		cv::remap(input, output, rmap[1][0], rmap[1][1], cv::INTER_LINEAR);
	}
	else if (input.cols == image_size_dwn.width)
	{
		cv::remap(input, output, dwnrmap[1][0], dwnrmap[1][1], cv::INTER_LINEAR);
	}
	else
	{
		if (scale_dwn != 1)
			cv::remap(input, output, dwnrmap[1][0], dwnrmap[1][1], cv::INTER_LINEAR);
		else
			cv::remap(input, output, rmap[1][0], rmap[1][1], cv::INTER_LINEAR);
	}
	
	return output;
}

void StereoBase::getRectifiedImage(cv::Mat &iLeftImg, cv::Mat &iRightImg)
{
	cv::remap(leftImage, iLeftImg, rmap[0][0], rmap[0][1], cv::INTER_LINEAR);
	cv::remap(rightImage, iRightImg, rmap[1][0], rmap[1][1], cv::INTER_LINEAR);
}


void StereoBase::getDownRectifiedImage(cv::Mat &oLeftImg, cv::Mat &oRightImg)
{
	cv::remap(dwnleftImage, oLeftImg, dwnrmap[0][0], dwnrmap[0][1], cv::INTER_LINEAR);
	cv::remap(dwnrightImage, oRightImg, dwnrmap[1][0], dwnrmap[1][1], cv::INTER_LINEAR);
}


void StereoBase::writecvPoint3fToPLYFile(const std::vector<cv::Point3f> &points, 
	const std::string &filename)
{

	std::ofstream outFile (filename.c_str());
	if (!outFile)
	{
		std::cerr << "Error opening output file: " << filename << std::endl;
		exit(1);
	}
	/************************************************************************/
	/* Header                                                               */
	/************************************************************************/
	const int pointNum = points.size();
	int valid_point_count = 0;
	const int triangleNum = 0;

	outFile << "ply" << std::endl;
	outFile << "format ascii 1.0" << std::endl;
	outFile << "element vertex ";
	long pos = outFile.tellp();
	outFile << "           " <<std::endl;
	outFile << "property float x" << std::endl;
	outFile << "property float y" << std::endl;
	outFile << "property float z" << std::endl;
	outFile << "property uchar red" << std::endl;
	outFile << "property uchar green" << std::endl;
	outFile << "property uchar blue" << std::endl;

	outFile << "end_header" << std::endl;

	/************************************************************************/
	/* Points                                                               */
	/************************************************************************/
	for (int i = 0; i < pointNum; i++)
	{
		outFile << points[i].x << " ";
		outFile << points[i].y<< " ";
		outFile << points[i].z << " ";

		// RGB
		outFile << 85 << " ";
		outFile << 176 << " ";
		outFile << 63 << " ";

		outFile << std::endl;
		valid_point_count++;
	}

	outFile.seekp (pos);
	outFile << valid_point_count;
	outFile.close();
	return;
}

void StereoBase::writePointCloudToPLYFile(const cv::Mat &cloud, 
	const cv::Mat &normals, 
	const cv::Mat &color,
	const std::string &filename)
{
	const bool hasNormal = !normals.empty();
	const bool hasColor = !color.empty();

	if (cloud.size() != normals.size() && hasNormal)
	{
		std::cerr << "Input cloud size must same as normal size" << std::endl;
		exit(1);
	}
	if (cloud.size() != color.size() && hasColor)
	{
		std::cerr << "Input cloud size must same as color size" << std::endl;
		exit(1);
	}

	std::ofstream outFile (filename.c_str());
    if (!outFile.is_open())
	{
		std::cerr << "Error opening output file: " << filename << std::endl;
		exit(1);
	}

	/************************************************************************/
	/* Header                                                               */
	/************************************************************************/
	const int pointNum = (int) cloud.rows * cloud.cols;
	int valid_point_count = 0;
	const int triangleNum = 0;

	outFile << "ply" << std::endl;
	outFile << "format ascii 1.0" << std::endl;
	outFile << "element vertex ";
	long pos = outFile.tellp();
	outFile << "           " <<std::endl;
	outFile << "property float x" << std::endl;
	outFile << "property float y" << std::endl;
	outFile << "property float z" << std::endl;
	if (hasNormal)
	{
		outFile << "property float nx" << std::endl;
		outFile << "property float ny" << std::endl;
		outFile << "property float nz" << std::endl;
	}	
	if (hasColor)
	{
		outFile << "property uchar red" << std::endl;
		outFile << "property uchar green" << std::endl;
		outFile << "property uchar blue" << std::endl;
	}
	// 	outFile << "element face " << triangleNum << endl;
	// 	outFile << "property list uchar int vertex_index" << endl;
	outFile << "end_header" << std::endl;

	/************************************************************************/
	/* Points                                                               */
	/************************************************************************/
	for (int y = 0; y < cloud.rows; y++)
	{
		for (int x = 0; x < cloud.cols; x++)
		{
			cv::Point3f p = cloud.at<cv::Vec3f>(y, x);
			if (p.z > 5000)
            {
                p.x = 0;
                p.y = 0;
                p.z = 0;
            }
			outFile << p.x << " ";
			outFile << p.y<< " ";
			outFile << p.z << " ";

			if (hasNormal)
			{
				cv::Point3f n = normals.at<cv::Vec3f>(y, x);
				outFile << n.x << " ";
				outFile << n.y<< " ";
				outFile << n.z << " ";
			}
			if (hasColor)
			{
				cv::Vec3b c = color.at<cv::Vec3b>(y, x);
				// BGR
				outFile << (int)c[2] << " ";
				outFile << (int)c[1] << " ";
				outFile << (int)c[0] << " ";
			}

			outFile << std::endl;
			valid_point_count++;
		}
	}

	/************************************************************************/
	/* Triangles                                                            */
	/************************************************************************/
	// 	for ( int ti = 0; ti < triangleNum; ++ti )
	// 	{
	// 		const Triangle& triangle = triangleVec[ ti ];
	// 		outFile << "3 ";
	// 
	// 		for ( int vi = 0; vi < 3; ++vi )
	// 			outFile << triangle._v[ vi ] << " ";
	// 
	// 		outFile << endl;
	// 	}

	outFile.seekp (pos);
	outFile << valid_point_count;
	outFile.close();
	return;
}
