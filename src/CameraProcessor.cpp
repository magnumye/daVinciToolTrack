#include "simulator.h"
#include "CameraProcessor.h"

#include "ShapeContextPro.h"

void decompose_rotation_xyz(const cv::Mat &R, double& thetaX, 
							double& thetaY, double& thetaZ);

void compose_rotation(const double &thetaX, const double &thetaY,
					  const double &thetaZ, cv::Mat &R);





CameraProcessor::CameraProcessor(double fx, double fy, double px, double py, int w, int h,
								   const cv::Mat& cHw_, const cv::Mat& cHw2_,
								   const std::vector<cv::Scalar>& LND_partcolors,
								   const std::map<std::string, int>& LND_partname2ids): Fx(fx), Fy(fy), Px(px), Py(py), width(w), height(h)
{

	tool_partcolors.resize(LND_partcolors.size());
	std::copy(LND_partcolors.begin(), LND_partcolors.end(), tool_partcolors.begin());

	tool_partname2ids.insert(LND_partname2ids.begin(), LND_partname2ids.end());

	qgo_detector = new QGODetector();

	ekFilter_psm1 = NULL;
	ekFilter_psm2 = NULL;

	camera_img_id = 0;

	cHw_.copyTo(cHw);
	cHw2_.copyTo(cHw2);
	corr_T = cv::Mat::eye(4, 4, CV_64F);
	corr_T2 = cv::Mat::eye(4, 4, CV_64F);

	T_cam_need_set = false;
	T2_cam_need_set = false;
	corr_T_cam = cv::Mat::eye(4, 4, CV_64F);
	corr_T2_cam = cv::Mat::eye(4, 4, CV_64F);

	low_num_threshold_psm1 = 3;
	high_num_threshold_psm1= 6;
	low_num_threshold_psm2 = 3;
	high_num_threshold_psm2= 6;

	QObject::connect(&timer_, &QTimer::timeout, this, &CameraProcessor::process);
}

CameraProcessor::~CameraProcessor()
{
	if (qgo_detector != NULL)
		delete qgo_detector;
	if (ekFilter_psm1 != NULL)
		delete ekFilter_psm1;
	if (ekFilter_psm2 != NULL)
		delete ekFilter_psm2;

}

void CameraProcessor::set_simulator(Simulator* sim)
{
	simulator_ = sim;
}

void CameraProcessor::start()
{
	timer_.start(20);
	timer_.moveToThread(&thread_);
	this->moveToThread(&thread_);
	thread_.start();
}

void CameraProcessor::process() 
{
	///////////////////////////////////
	// From global to local
	cv::Mat render_img_flip_cv_local, camera_image_local;
	Eigen::Matrix<double, 4, 4, Eigen::RowMajor> psm1_bHe_local, psm1_bHj4_local, psm1_bHj5_local;
	Eigen::Matrix<double, 4, 4, Eigen::RowMajor> psm2_bHe_local, psm2_bHj4_local, psm2_bHj5_local;
	float psm1_jaw_local, psm2_jaw_local;
	std::vector<cv::Rect> psm1_part_boxes_local; 
	std::vector<std::string> psm1_class_names_local;
	std::vector<cv::Point2f> psm1_projectedKeypoints_local;
	std::vector<cv::Point3f> psm1_allKeypoints_no_corr_local;
	std::vector<cv::Rect> psm2_part_boxes_local;
	std::vector<std::string> psm2_class_names_local;
	std::vector<cv::Point2f> psm2_projectedKeypoints_local;
	std::vector<cv::Point3f> psm2_allKeypoints_no_corr_local;
	std::vector<int> psm1_template_half_sizes_local;
	std::vector<int> psm2_template_half_sizes_local;
	float psm1_slope_local;
	float psm2_slope_local;

	if(!(simulator_->ReadVirtualRenderingOutput(render_img_flip_cv_local, camera_image_local,
		psm1_part_boxes_local, 
		psm1_class_names_local, psm1_projectedKeypoints_local,
		psm1_allKeypoints_no_corr_local,
		psm1_template_half_sizes_local,
		psm2_part_boxes_local, 
		psm2_class_names_local, psm2_projectedKeypoints_local,
		psm2_allKeypoints_no_corr_local,
		psm2_template_half_sizes_local, 
		psm1_bHe_local,
		psm1_bHj4_local,
		psm1_bHj5_local,
		psm2_bHe_local,
		psm2_bHj4_local,
		psm2_bHj5_local,
		psm1_jaw_local, psm2_jaw_local,
		psm1_slope_local, psm2_slope_local))) return;
	// Add templates to QGO detector.
	cv::Mat quantized_angle;
	cv::Mat magnitude;
	std::vector<cv::Rect> psm1_bounding_boxes, psm2_bounding_boxes;
	std::vector<bool> psm1_states, psm2_states;
	qgo_detector->addDualTemplateSet(render_img_flip_cv_local, 
		psm2_part_boxes_local, psm2_class_names_local,
		psm1_part_boxes_local, psm1_class_names_local, 
		quantized_angle, magnitude, 
		psm2_bounding_boxes, psm1_bounding_boxes,
		psm2_states, psm1_states);


	// Perform matching using available templates.
	bool T_cam_set = false;
	bool T2_cam_set = false;

	process_camera(psm1_allKeypoints_no_corr_local, psm1_projectedKeypoints_local,
		psm2_allKeypoints_no_corr_local, psm2_projectedKeypoints_local,
		psm1_template_half_sizes_local,
		psm2_template_half_sizes_local,
		camera_image_local,
		corr_T, corr_T2,
		T_cam_set, T2_cam_set, camera_img_id);

	if (T_cam_set)
	{
		T_cam_need_set = true;
		corr_T.copyTo(corr_T_cam);
	}
	if (T2_cam_set)
	{
		T2_cam_need_set = true;
		corr_T2.copyTo(corr_T2_cam);
	}
	simulator_->WriteCameraOutput(T_cam_need_set, T2_cam_need_set, corr_T_cam, corr_T2_cam);



	camera_img_id++;

	draw_skel(camera_image_local, cHw, psm1_bHj4_local, psm1_bHj5_local, 
		psm1_bHe_local, psm1_jaw_local, false, 1);
	draw_skel(camera_image_local, cHw, psm1_bHj4_local, psm1_bHj5_local, 
		psm1_bHe_local, psm1_jaw_local, corr_T, 1, psm1_slope_local);

	draw_skel(camera_image_local, cHw2, psm2_bHj4_local, psm2_bHj5_local, 
		psm2_bHe_local, psm2_jaw_local, false, 2);
	draw_skel(camera_image_local, cHw2, psm2_bHj4_local, psm2_bHj5_local, 
		psm2_bHe_local, psm2_jaw_local, corr_T2, 2, psm2_slope_local);

#ifdef __linux__    // Linux ROS specific
	// Framerate calculation
	timeval t_curr;
	gettimeofday(&t_curr, NULL);
	time_count.push_back(t_curr);
	//    if (time_count.size() > 100)
	{
		//        time_count.pop_front();
		double td = time_diff(&time_count.front(), &time_count.back());
		double fps = time_count.size()/td;
		std::stringstream str;
		str << setprecision(4) << fps;
		Q_EMIT dispFrameRate("FPS: " + QString::number(fps));
		cv::putText(camera_image_local, "FPS: "+str.str(), cv::Point(5,15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0) );
	}
#endif


#ifdef _WIN32

	time_t t_curr;
	time(&t_curr);
	time_count.push_back(t_curr);
	if (time_count.size() > 100)
	{
		time_count.pop_front();
		double td = difftime(time_count.back(), time_count.front());
		Q_EMIT dispFrameRate("FPS: " + QString::number(100.0/td));
	}
#endif

	QImage qtemp_camera = cvtCvMat2QImage(camera_image_local);

	Q_EMIT updateCameraImage(qtemp_camera);
}

bool CameraProcessor::process_camera(const std::vector<cv::Point3f>& psm1_allKeypoints,
									  const std::vector<cv::Point2f>& psm1_projectedKeypoints,
									  const std::vector<cv::Point3f>& psm2_allKeypoints,
									  const std::vector<cv::Point2f>& psm2_projectedKeypoints,
									  const std::vector<int>& ra_template_half_sizes,
									  const std::vector<int>& la_template_half_sizes,
									  cv::Mat& camera_image, cv::Mat& psm1_err_T,
									  cv::Mat& psm2_err_T, bool& T_cam_set,
									  bool& T2_cam_set, int img_id)
{

	cv::Mat rectify_left;
	camera_image.copyTo(rectify_left);


	std::vector<Match> psm1_matches, psm2_matches;
	std::vector<std::string> psm1_class_ids, psm2_class_ids;
	cv::Mat quantized_angles, quantized_display, magnitude;

	qgo_detector->match(rectify_left, 30, psm2_matches, psm2_class_ids, 
		psm1_matches, psm1_class_ids, quantized_angles, magnitude, img_id);
	//std::cout << matches.size() << " " << similarities.size() << std::endl;

	//qgo_detector->colormap(quantized_angles, quantized_display);
	//quantized_display.copyTo(camera_image);



	bool psm1_shaft_visible = false;
	bool psm2_shaft_visible = false;
	cv::Point psm1_shaft_centre, psm2_shaft_centre;

	// PSM1
	{
		std::vector<cv::Point2f> verifiedCamerapoints_t;
		std::vector<std::string> verifiedNames_t;
		if (psm1_matches.size() > 0)
			shape_context_verification(camera_image, psm1_matches, psm1_projectedKeypoints, 
			verifiedCamerapoints_t, verifiedNames_t, ra_template_half_sizes);
		std::cout << "PSM1: " << verifiedCamerapoints_t.size() << std::endl; 
		std::vector<cv::Point2f> verifiedCamerapoints;
		std::vector<std::string> verifiedNames;
		for (int ci = 0; ci < verifiedCamerapoints_t.size(); ci++)
		{
			if (verifiedNames_t[ci] != "shaft_centre")
			{
				verifiedCamerapoints.push_back(verifiedCamerapoints_t[ci]);
				verifiedNames.push_back(verifiedNames_t[ci]);

			}
			else
			{
				psm1_shaft_visible = true;
				verifiedCamerapoints.push_back(verifiedCamerapoints_t[ci]);
				verifiedNames.push_back(verifiedNames_t[ci]);
				psm1_shaft_centre = verifiedCamerapoints_t[ci];
			}

		}



		// PSM1 shaft centre lin detection
		int x_limit = camera_image.cols/2; 
		int y_limit = 20; 
		int width_limit = camera_image.cols - x_limit;
		int height_limit = camera_image.rows - 2*y_limit;
		cv::Rect roi = cv::Rect(x_limit, y_limit, width_limit, height_limit);
		//detect_shaft(camera_image, binary_mask, roi);

		if ((int)verifiedCamerapoints.size() > low_num_threshold_psm1 || img_id == 0)
		{
			// Solve pnp
			cv::Mat rvec, tvec;
			bool pose_good = false;
			if (ekFilter_psm1 == NULL && (int)verifiedCamerapoints.size() > high_num_threshold_psm1)
			{
				cv::Mat subP = cv::Mat::eye(3, 3, CV_32FC1);
				subP.at<float>(0,0) = Fx; subP.at<float>(1,1) = Fy;
				subP.at<float>(0,2) = Px; subP.at<float>(1,2) = Py;
				cv::Mat distCoeffs;
				std::vector<cv::Point3f> model3dPoints;
				for (int ci = 0; ci < verifiedCamerapoints.size(); ci++)
				{
					model3dPoints.push_back(psm1_allKeypoints[tool_partname2ids[verifiedNames[ci]]]);
				}


				pose_good = cv::solvePnP(model3dPoints, verifiedCamerapoints, subP, distCoeffs, rvec, tvec, false, CV_EPNP);
				//std::cout << tvec << std::endl;
				//cv::Mat inliers; pose_good = true;
				//cv::solvePnPRansac(model3dPoints, verifiedCamerapoints, subP, distCoeffs, rvec, tvec, false, 100, 3.0, 7, inliers, CV_ITERATIVE);
				std::vector<cv::Point2f> projected2dPoints;
				if (pose_good)
				{
					cv::projectPoints(psm1_allKeypoints, rvec, tvec, subP, distCoeffs, projected2dPoints);
				}

				// Experimental
				if (!psm1_shaft_visible) pose_good = false; // if shaft centre not exist, don't trust
			}
			if (ekFilter_psm1 == NULL)
			{
				if (pose_good)
				{
					int state_dimension = 6, measure_dimension = psm1_allKeypoints.size()*2;
					cv::Mat init_state = cv::Mat::zeros(state_dimension, 1, CV_64F);
					cv::Mat rot_mat;
					cv::Rodrigues(rvec, rot_mat);
					double theta_x, theta_y, theta_z;
					decompose_rotation_xyz(rot_mat, theta_x, theta_y, theta_z);
					init_state.at<double>(0,0) = theta_x;
					init_state.at<double>(1,0) = theta_y;
					init_state.at<double>(2,0) = theta_z;
					init_state.at<double>(3,0) = tvec.at<double>(0,0);
					init_state.at<double>(4,0) = tvec.at<double>(1,0);
					init_state.at<double>(5,0) = tvec.at<double>(2,0);
					initEKFPSM1(state_dimension, measure_dimension, state_dimension, measure_dimension, init_state);

					cv::Mat temp = psm1_err_T(cv::Rect(0,0,3,3));
					compose_rotation(theta_x, theta_y, theta_z, temp);
					psm1_err_T.at<double>(0,3) = tvec.at<double>(0,0);
					psm1_err_T.at<double>(1,3) = tvec.at<double>(1,0);
					psm1_err_T.at<double>(2,3) = tvec.at<double>(2,0);
					std::cout<< psm1_err_T << std::endl;
					T_cam_set = true;
				}
			}
			else// EKF process
			{
				// Step 1. Estimate state
				ekFilter_psm1->Calc_x_estimated();
				// Step 2. Estiamte error covariance
				ekFilter_psm1->Calc_P_estimated();
				// Step 3. Compute Kalman gain
				std::vector<double> x_estimate;
				ekFilter_psm1->Get_x_estimated(x_estimate);
				cv::Mat H, z_estimated;
				calc_H_z_estiamted(psm1_allKeypoints, x_estimate, Fx, Fy,
					Px, Py, H, z_estimated);
				Eigen::MatrixXd H_ = cvMat2Eigen(H);
				ekFilter_psm1->Set_H(H_);
				ekFilter_psm1->Calc_S();
				ekFilter_psm1->Calc_K();
				// Step 4. Update estimate with measurement
				cv::Mat z = cv::Mat::zeros(psm1_allKeypoints.size()*2, 1, CV_64F);

				std::vector<bool> z_valid(psm1_allKeypoints.size()*2, false);
				//assert(psm1_allKeypoints.size() == verifiedCamerapoints_t.size());
				for (int ci = 0; ci < verifiedCamerapoints_t.size(); ci++)
				{
					int vid = tool_partname2ids[verifiedNames_t[ci]] * 2;
					z_valid[vid] = true;
					z_valid[vid+1] = true;
					z.at<double>(vid, 0) = verifiedCamerapoints_t[ci].x;
					z.at<double>(vid+1, 0) = verifiedCamerapoints_t[ci].y;
				}
				//std::cout << "z_measured" << std::endl;
				//std::cout << z << std::endl;

				Eigen::MatrixXd z_ = cvMat2Eigen(z), z_estimated_ = cvMat2Eigen(z_estimated);
				ekFilter_psm1->Set_z(z_);
				ekFilter_psm1->Set_z_estimated(z_estimated_);
				//ekFilter->Calc_x_corrected();
				ekFilter_psm1->Calc_x_corrected(z_valid);
				std::vector<double> x_corrected;
				ekFilter_psm1->Get_x_corrected(x_corrected);
				// Step 5. Update error covariance
				ekFilter_psm1->Calc_P_corrected();
				// Step 6. Finish for next loop
				ekFilter_psm1->Finish_for_next();

				cv::Mat temp = psm1_err_T(cv::Rect(0,0,3,3));
				compose_rotation(x_corrected[0], x_corrected[1], x_corrected[2], temp);
				psm1_err_T.at<double>(0,3) = x_corrected[3];
				psm1_err_T.at<double>(1,3) = x_corrected[4];
				psm1_err_T.at<double>(2,3) = x_corrected[5];
			}
		}
	}
	// PSM2
	{
		std::vector<cv::Point2f> verifiedCamerapoints_t;
		std::vector<std::string> verifiedNames_t;
		if (psm2_matches.size() > 0)
			shape_context_verification(camera_image, psm2_matches, psm2_projectedKeypoints, 
			verifiedCamerapoints_t, verifiedNames_t, la_template_half_sizes);
		std::cout << "PSM2: " << verifiedCamerapoints_t.size() << std::endl; 
		std::vector<cv::Point2f> verifiedCamerapoints;
		std::vector<std::string> verifiedNames;
		for (int ci = 0; ci < verifiedCamerapoints_t.size(); ci++)
		{
			if (verifiedNames_t[ci] != "shaft_centre")
			{
				verifiedCamerapoints.push_back(verifiedCamerapoints_t[ci]);
				verifiedNames.push_back(verifiedNames_t[ci]);

			}
			else
			{
				verifiedCamerapoints.push_back(verifiedCamerapoints_t[ci]);
				verifiedNames.push_back(verifiedNames_t[ci]);
				psm2_shaft_visible = true;
				psm2_shaft_centre = verifiedCamerapoints_t[ci];
			}
		}


		// PSM2 shaft centre lin detection
		int x_limit = 0; 
		int y_limit = 20; 
		int width_limit = camera_image.cols/2;
		int height_limit = camera_image.rows - 2*y_limit;
		cv::Rect roi = cv::Rect(x_limit, y_limit, width_limit, height_limit);
		//detect_shaft(camera_image, binary_mask, roi);

		if ((int)verifiedCamerapoints.size() > low_num_threshold_psm2 || img_id == 0)
		{
			// Solve pnp
			cv::Mat rvec, tvec;
			bool pose_good = false;
			if (ekFilter_psm2 == NULL && (int)verifiedCamerapoints.size() > high_num_threshold_psm2)
			{
				cv::Mat subP = cv::Mat::eye(3, 3, CV_32FC1);
				subP.at<float>(0,0) = Fx; subP.at<float>(1,1) = Fy;
				subP.at<float>(0,2) = Px; subP.at<float>(1,2) = Py;
				cv::Mat distCoeffs;
				std::vector<cv::Point3f> model3dPoints;
				for (int ci = 0; ci < verifiedCamerapoints.size(); ci++)
				{
					model3dPoints.push_back(psm2_allKeypoints[tool_partname2ids[verifiedNames[ci]]]);
				}

				pose_good = cv::solvePnP(model3dPoints, verifiedCamerapoints, subP, distCoeffs, rvec, tvec, false, CV_EPNP);
				//std::cout << tvec << std::endl;
				//cv::Mat inliers; pose_good = true;
				//cv::solvePnPRansac(model3dPoints, verifiedCamerapoints, subP, distCoeffs, rvec, tvec, false, 100, 3.0, 7, inliers, CV_ITERATIVE);
				std::vector<cv::Point2f> projected2dPoints;
				if (pose_good)
				{
					cv::projectPoints(psm2_allKeypoints, rvec, tvec, subP, distCoeffs, projected2dPoints);
				}

				// Experimental
				if (!psm2_shaft_visible) pose_good = false; // if shaft centre not exist, don't trust
			}
			if (ekFilter_psm2 == NULL)
			{

				if (pose_good)
				{
					int state_dimension = 6, measure_dimension = psm2_allKeypoints.size()*2;
					cv::Mat init_state = cv::Mat::zeros(state_dimension, 1, CV_64F);
					cv::Mat rot_mat;
					cv::Rodrigues(rvec, rot_mat);
					double theta_x, theta_y, theta_z;
					decompose_rotation_xyz(rot_mat, theta_x, theta_y, theta_z);
					init_state.at<double>(0,0) = theta_x;
					init_state.at<double>(1,0) = theta_y;
					init_state.at<double>(2,0) = theta_z;
					init_state.at<double>(3,0) = tvec.at<double>(0,0);
					init_state.at<double>(4,0) = tvec.at<double>(1,0);
					init_state.at<double>(5,0) = tvec.at<double>(2,0);
					initEKFPSM2(state_dimension, measure_dimension, state_dimension, measure_dimension, init_state);
					cv::Mat temp = psm2_err_T(cv::Rect(0,0,3,3));
					compose_rotation(theta_x, theta_y, theta_z, temp);
					psm2_err_T.at<double>(0,3) = tvec.at<double>(0,0);
					psm2_err_T.at<double>(1,3) = tvec.at<double>(1,0);
					psm2_err_T.at<double>(2,3) = tvec.at<double>(2,0);
					std::cout<< psm2_err_T << std::endl;
					T2_cam_set = true;
				}
			}
			else// EKF process
			{
				// Step 1. Estimate state
				ekFilter_psm2->Calc_x_estimated();
				// Step 2. Estiamte error covariance
				ekFilter_psm2->Calc_P_estimated();
				// Step 3. Compute Kalman gain
				std::vector<double> x_estimate;
				ekFilter_psm2->Get_x_estimated(x_estimate);
				//std::cout << "x_estimate: " << x_estimate[3] << "," << x_estimate[4] << "," << x_estimate[5] << std::endl;
				cv::Mat H, z_estimated;
				calc_H_z_estiamted(psm2_allKeypoints, x_estimate, Fx, Fy,
					Px, Py, H, z_estimated);
				Eigen::MatrixXd H_ = cvMat2Eigen(H);
				ekFilter_psm2->Set_H(H_);
				ekFilter_psm2->Calc_S();
				ekFilter_psm2->Calc_K();
				// Step 4. Update estimate with measurement
				cv::Mat z = cv::Mat::zeros(psm2_allKeypoints.size()*2, 1, CV_64F);
				//assert(psm2_allKeypoints.size() == verifiedCamerapoints_t.size());
				std::vector<bool> z_valid(psm2_allKeypoints.size()*2, false);
				for (int ci = 0; ci < verifiedCamerapoints_t.size(); ci++)
				{
					int vid = tool_partname2ids[verifiedNames_t[ci]] * 2;
					z_valid[vid] = true;
					z_valid[vid+1] = true;
					z.at<double>(vid, 0) = verifiedCamerapoints_t[ci].x;
					z.at<double>(vid+1, 0) = verifiedCamerapoints_t[ci].y;
				}
				//std::cout << "z_measured" << std::endl;
				//std::cout << z << std::endl;
				//std::cout << "z_estimated" << std::endl;
				//std::cout << z_estimated << std::endl;
				Eigen::MatrixXd z_ = cvMat2Eigen(z), z_estimated_ = cvMat2Eigen(z_estimated);
				ekFilter_psm2->Set_z(z_);
				ekFilter_psm2->Set_z_estimated(z_estimated_);
				//ekFilter->Calc_x_corrected();
				ekFilter_psm2->Calc_x_corrected(z_valid);
				std::vector<double> x_corrected;
				ekFilter_psm2->Get_x_corrected(x_corrected);
				// Step 5. Update error covariance
				ekFilter_psm2->Calc_P_corrected();
				// Step 6. Finish for next loop
				ekFilter_psm2->Finish_for_next();
				cv::Mat temp = psm2_err_T(cv::Rect(0,0,3,3));
				compose_rotation(x_corrected[0], x_corrected[1], x_corrected[2], temp);
				//std::cout << "x_corrected: " << x_corrected[3] << "," << x_corrected[4] << "," << x_corrected[5] << std::endl;
				psm2_err_T.at<double>(0,3) = x_corrected[3];
				psm2_err_T.at<double>(1,3) = x_corrected[4];
				psm2_err_T.at<double>(2,3) = x_corrected[5];
			}
		}
	}


	return true;
}



void CameraProcessor::shape_context_verification(cv::Mat& img, const std::vector<Match>& matches, 
												  const std::vector<cv::Point2f>& projectedKeypoints, 
												  std::vector<cv::Point2f>& verifiedCamerapoints,
												  std::vector<std::string>& verifiedNames,
												  const std::vector<int>& template_half_sizes)
{

	// Shape Context Verification
	std::vector<ProsacPoint2f> pointsA;
	std::vector<ProsacPoint2f> pointsB;
	pointsA.reserve(matches. size());
	pointsB.reserve(matches.size());
	std::vector<double> scores;
	scores.reserve(matches.size());
	std::vector<int> landmarkIds;
	landmarkIds.reserve(matches.size());
	std::vector<int> detectionIds;
	detectionIds.reserve(matches.size());

	for (int mh = 0; mh < matches.size(); mh++)
	{

		ProsacPoint2f pt1 = {projectedKeypoints[tool_partname2ids[matches[mh].class_id]].x, 
			projectedKeypoints[tool_partname2ids[matches[mh].class_id]].y};
		ProsacPoint2f pt2 = {matches[mh].x+template_half_sizes[tool_partname2ids[matches[mh].class_id]],
			matches[mh].y+template_half_sizes[tool_partname2ids[matches[mh].class_id]]};


		pointsA.push_back(pt1);
		pointsB.push_back(pt2);
		scores.push_back(matches[mh].similarity);
		landmarkIds.push_back(tool_partname2ids[matches[mh].class_id]);
		detectionIds.push_back(mh);
	}

	SCConfig config = {2, 500};
	ShapeContextPro	scp(config);
	scp.SetData(&pointsA, &pointsB, &landmarkIds);
	scp.SetDetectionIds(&detectionIds);
	scp.SetScores(&scores);
	scp.ComputeModel();

	const int& numInliers = scp.GetBestNumInliers();
	std::vector<bool> inliers;

	if (numInliers > 3)
	{
		inliers = scp.GetBestInliers();
		for (int il = 0; il < inliers.size(); il++)
		{
			if (inliers[il])
			{
				cv::Point2f pt(matches[il].x+template_half_sizes[tool_partname2ids[matches[il].class_id]], 
					matches[il].y+template_half_sizes[tool_partname2ids[matches[il].class_id]]);

				verifiedCamerapoints.push_back(pt);
				verifiedNames.push_back(matches[il].class_id);


			}
		}
	}

	//std::cout << scp.GetBestNumInliers() << std::endl;
}

void CameraProcessor::calc_H_z_estiamted(const std::vector<cv::Point3f>& kpts_cstar, 
										  const std::vector<double>& state_estimated, 
										  double f_x, double f_y, double d_x, double d_y,
										  cv::Mat& H, cv::Mat& z_estimated)
{
	double theta_x = state_estimated[0];
	double theta_y = state_estimated[1];
	double theta_z = state_estimated[2];
	double t_x = state_estimated[3];
	double t_y = state_estimated[4];
	double t_z = state_estimated[5];

	double sx = sin(theta_x), sy = sin(theta_y), sz = sin(theta_z),
		cx = cos(theta_x), cy = cos(theta_y), cz = cos(theta_z);

	H = cv::Mat::zeros(2*(int)kpts_cstar.size(), (int)state_estimated.size(), CV_64F);
	z_estimated = cv::Mat::zeros(2*(int)kpts_cstar.size(), 1, CV_64F);
	for (int k = 0; k < kpts_cstar.size(); k++)
	{
		double c = kpts_cstar[k].x;
		double r = kpts_cstar[k].y;
		double m = kpts_cstar[k].z;

		double F = c*cy*cz - r*cy*sz + m*sy + t_x;
		double L = c*sx*sy*cz + c*cx*sz - r*sx*sy*sz + r*cx*cz - m*sx*cy + t_y;
		double G = -c*cx*sy*cz + c*sx*sz + r*cx*sy*sz + r*sx*cz + m*cx*cy + t_z;
		double G_2 = G*G;
		/////////// Calc z_estimated
		z_estimated.at<double>(2*k,0) = f_x*(F/G) + d_x; z_estimated.at<double>(2*k+1,0) = f_y*(L/G) + d_y;

		/////////// Calc H
		///------------
		double F_over_theta_x = 0;

		double G_over_theta_x = c*sx*sy*cz + c*cx*sz - r*sx*sy*sz + r*cx*cz - m*sx*cy;

		double u_over_theta_x = f_x*((F_over_theta_x*G - G_over_theta_x*F)/G_2);

		///------------
		double F_over_theta_y = -c*sy*cz + r*sy*sz + m*cy;

		double G_over_theta_y = -c*cx*cy*cz + r*cx*cy*sz - m*cx*sy;

		double u_over_theta_y = f_x*((F_over_theta_y*G - G_over_theta_y*F)/G_2);


		///------------
		double F_over_theta_z = -c*cy*sz - r*cy*cz;

		double G_over_theta_z = c*cx*sy*sz + c*sx*cz + r*cx*sy*cz - r*sx*sz;

		double u_over_theta_z = f_x*((F_over_theta_z*G - G_over_theta_z*F)/G_2);

		///------------
		double u_over_t_x = f_x/G;

		///------------
		double u_over_t_y = 0;

		///------------
		double u_over_t_z = -f_x*(F/G_2);

		//////////////////

		///------------
		double L_over_theta_x = c*cx*sy*cz - c*sx*sz - r*cx*sy*sz - r*sx*cz - m*cx*cy; 

		double v_over_theta_x = f_y*((L_over_theta_x*G - G_over_theta_x*L)/G_2);

		///------------
		double L_over_theta_y = c*sx*cy*cz - r*sx*cy*sz + m*sx*sy;

		double v_over_theta_y = f_y*((L_over_theta_y*G - G_over_theta_y*L)/G_2);

		///------------
		double L_over_theta_z = -c*sx*sy*sz + c*cx*cz - r*sx*sy*cz - r*cx*sz;

		double v_over_theta_z = f_y*((L_over_theta_z*G - G_over_theta_z*L)/G_2);

		///------------
		double v_over_t_x = 0;

		///------------
		double v_over_t_y = f_y/G;

		///------------
		double v_over_t_z = -f_y*(L/G_2);

		/////////////////
		int rows = k*2;
		H.at<double>(rows, 0) = u_over_theta_x; H.at<double>(rows, 1) = u_over_theta_y; 
		H.at<double>(rows, 2) = u_over_theta_z; H.at<double>(rows, 3) = u_over_t_x; 
		H.at<double>(rows, 4) = u_over_t_y; H.at<double>(rows, 5) = u_over_t_z;
		rows = rows + 1;
		H.at<double>(rows, 0) = v_over_theta_x; H.at<double>(rows, 1) = v_over_theta_y; 
		H.at<double>(rows, 2) = v_over_theta_z; H.at<double>(rows, 3) = v_over_t_x; 
		H.at<double>(rows, 4) = v_over_t_y; H.at<double>(rows, 5) = v_over_t_z;
	}

	//

	//std::cout << "z_estimated from h(.)" << std::endl;
	//std::cout << z_estimated << std::endl;
	//cv::Mat state_estimated_M = cv::Mat::zeros(state_estimated.size(), 1, CV_64F);
	//state_estimated_M.at<double>(0,0) = state_estimated[0];
	//state_estimated_M.at<double>(1,0) = state_estimated[1];
	//state_estimated_M.at<double>(2,0) = state_estimated[2];
	//state_estimated_M.at<double>(3,0) = state_estimated[3];
	//state_estimated_M.at<double>(4,0) = state_estimated[4];
	//state_estimated_M.at<double>(5,0) = state_estimated[5];
	//std::cout << "Jacobian H" << std::endl;
	//std::cout << H << std::endl;
	//std::cout << "state_estimate" << std::endl;
	//std::cout << state_estimated_M << std::endl;
	//z_estimated = H * state_estimated_M;
	//std::cout << "z_estimated from H.x" << std::endl;
	//std::cout << z_estimated << std::endl;
}

Eigen::MatrixXd CameraProcessor::cvMat2Eigen(cv::Mat& m)
{
	Eigen::MatrixXd eigen_mat = Eigen::MatrixXd::Zero(m.rows, m.cols);
	for (auto i = 0; i < m.cols; i++)
	{
		for (auto j = 0; j < m.rows; j++)
		{
			eigen_mat(j, i) = m.at<double>(j, i);
		}
	}

	return eigen_mat;
}

void CameraProcessor::initEKFPSM1(int state_dim, int measure_dim, int w_dim, int v_dim, cv::Mat& init_state)
{
	ekFilter_psm1 = new ExtendedKalmanFilter();
	ekFilter_psm1->SetDimensions(state_dim, state_dim, measure_dim, w_dim, v_dim);
	ekFilter_psm1->InitMatrices();

	// Set initial state [theta_x, theta_y, theta_z, t_x, t_y, t_z].
	Eigen::MatrixXd x = cvMat2Eigen(init_state);
	ekFilter_psm1->Init_x(x);

	// Noises
	Eigen::MatrixXd Q;
	Q.setIdentity(state_dim, w_dim);
	Q = Q*0.1;
	Eigen::MatrixXd R;
	R.setIdentity(measure_dim, v_dim);
	R = R*0.1;

	ekFilter_psm1->Set_NoiseCovariance(Q, R);
	// Error covariance
	ekFilter_psm1->Init_P();
	// Jacobians: A, W, V. H should be calculated on-the-fly.
	ekFilter_psm1->Set_A();
	ekFilter_psm1->Set_W();
	ekFilter_psm1->Set_V();
}

void CameraProcessor::initEKFPSM2(int state_dim, int measure_dim, int w_dim, int v_dim, cv::Mat& init_state)
{
	ekFilter_psm2 = new ExtendedKalmanFilter();
	ekFilter_psm2->SetDimensions(state_dim, state_dim, measure_dim, w_dim, v_dim);
	ekFilter_psm2->InitMatrices();

	// Set initial state [theta_x, theta_y, theta_z, t_x, t_y, t_z].
	Eigen::MatrixXd x = cvMat2Eigen(init_state);
	ekFilter_psm2->Init_x(x);

	// Noises
	Eigen::MatrixXd Q;
	Q.setIdentity(state_dim, w_dim);
	Q = Q*0.1;
	Eigen::MatrixXd R;
	R.setIdentity(measure_dim, v_dim);
	R = R*0.1;

	ekFilter_psm2->Set_NoiseCovariance(Q, R);
	// Error covariance
	ekFilter_psm2->Init_P();
	// Jacobians: A, W, V. H should be calculated on-the-fly.
	ekFilter_psm2->Set_A();
	ekFilter_psm2->Set_W();
	ekFilter_psm2->Set_V();
}

void CameraProcessor::draw_skel(cv::Mat &img,
								 const cv::Mat &cHb,
								 const Eigen::Matrix<double, 4, 4, Eigen::RowMajor> &psm_Hj4,
								 const Eigen::Matrix<double, 4, 4, Eigen::RowMajor> &psm_Hj5,
								 const Eigen::Matrix<double, 4, 4, Eigen::RowMajor> &psm_He,
								 const float jaw_in, bool virt_or_cam, int psm_num)
{
	cv::Mat subP = cv::Mat::eye(3, 3, CV_32FC1);
	subP.at<float>(0,0) = Fx; subP.at<float>(1,1) = Fy;
	if (virt_or_cam)
	{
		//subP.at<float>(0,2) = m_cam->Px_win(); subP.at<float>(1,2) = m_cam->Py_win();
	}
	else
	{
		subP.at<float>(0,2) = Px; subP.at<float>(1,2) = Py;
	}


	std::vector<cv::Mat> bHj_vec;
	std::vector<cv::Point3f> cPj(7); // 1 shaft_far, 3 joint, 3 tips (cen/l/r)

	for (int i = 0; i < 3; i++)
		bHj_vec.push_back(cv::Mat(4, 4, CV_64F));


	::memcpy(bHj_vec[0].data, psm_Hj4.data(), sizeof(double)*16);
	::memcpy(bHj_vec[1].data, psm_Hj5.data(), sizeof(double)*16);
	::memcpy(bHj_vec[2].data, psm_He.data(), sizeof(double)*16);

	for (int i = 1; i <= bHj_vec.size(); i++)
	{
		// NB: bHj_vec unit in mm
		cv::Mat crHj_curr = cHb * bHj_vec[i-1];
		cPj[i].x = crHj_curr.at<double>(0, 3);
		cPj[i].y = crHj_curr.at<double>(1, 3);
		cPj[i].z = abs(crHj_curr.at<double>(2, 3));    // abs to ensure projection make sense
	}

	// NB: Shaft line
	cv::Mat j2H_offset = cv::Mat::eye(4,4,CV_64FC1);
	j2H_offset.at<double>(0, 3) = 50;  // 200mm
	cv::Mat temp = cHb * bHj_vec[0] * j2H_offset;
	cPj[0].x = temp.at<double>(0, 3);
	cPj[0].y = temp.at<double>(1, 3);
	cPj[0].z = abs(temp.at<double>(2, 3));    // abs to ensure projection make sense

	// Apply tip offset
	cv::Mat crHt;
	if (psm_num == 1)
		crHt = cHb * bHj_vec.back() * simulator_->psm1_tool->Tip_Mat();
	else if (psm_num == 2)
		crHt = cHb * bHj_vec.back() * simulator_->psm2_tool->Tip_Mat();

	cPj[4].x = crHt.at<double>(0, 3);
	cPj[4].y = crHt.at<double>(1, 3);
	cPj[4].z = crHt.at<double>(2, 3);

	/* Draw for extra points */
	std::vector<cv::Point3f> cHpkey = simulator_->calc_keypoints(bHj_vec[0], bHj_vec[1], bHj_vec[2], jaw_in, cHb, psm_num);


	cPj[5] = cHpkey[12];
	cPj[6] = cHpkey[13];

	// cPj[0]: shaft_far end
	// cPj[1]: shaft
	// cPj[2]: logo
	// cPj[3]: end-effector
	// cPj[4]: tip central
	// cPj[5]: tip flat (right)
	// cPj[6]: tip deep (left)

	std::vector<cv::Point2f> projectedPoints, projectedKeypoints;
	cv::Mat rVec, tVec, distCoeffs;
	rVec = cv::Mat::zeros(3,1,CV_32FC1); tVec = cv::Mat::zeros(3,1,CV_32FC1);
	cv::projectPoints(cPj, rVec, tVec, subP, distCoeffs, projectedPoints);
	cv::projectPoints(cHpkey, rVec, tVec, subP, distCoeffs, projectedKeypoints);

	cv::circle(img, projectedPoints[2], 5, cv::Scalar(0,0,200), -1);   // logo
	cv::circle(img, projectedPoints[3], 5, cv::Scalar(0,0,200), -1);   // jaw
	// cv::circle(img, projectedPoints[4], 5, cv::Scalar(0,0,200), -1);    // tip central
	cv::circle(img, projectedPoints[5], 3, cv::Scalar(0,0,200), -1);    // tip left (deep)
	cv::circle(img, projectedPoints[6], 3, cv::Scalar(0,0,200), -1);    // tip right (flat)

	float slope = (projectedPoints[1].y - projectedPoints[0].y) / (projectedPoints[1].x - projectedPoints[0].x);
	cv::Point p(0,0), q(img.cols,img.rows);
	if (psm_num == 2)//projectedPoints[0].x < projectedPoints[1].x)
	{
		p.y = -(projectedPoints[1].x - p.x) * slope + projectedPoints[1].y;
		cv::line(img, projectedPoints[1], p, cv::Scalar(0,0,200), 2);  // shaft -> shaft_far
	}
	else
	{
		q.y = -(projectedPoints[1].x - q.x) * slope + projectedPoints[1].y;
		cv::line(img, projectedPoints[1], q, cv::Scalar(0,0,200), 2);  // shaft -> shaft_far
	}

	cv::line(img, projectedPoints[3], projectedPoints[2], cv::Scalar(0,0,200), 2);  // end -> logo
	// cv::line(img, projectedPoints[4], projectedPoints[3], cv::Scalar(0,0,200), 2);  // tip -> jaw
	cv::line(img, projectedPoints[5], projectedPoints[3], cv::Scalar(0,0,200), 2);  // tip_flat -> jaw
	cv::line(img, projectedPoints[6], projectedPoints[3], cv::Scalar(0,0,200), 2);  // tip_deep -> jaw

	// cv::circle(img, projectedKeypoints[1], 3, cv::Scalar(0,255,255), -1);  // yellow shaft_pivot
	//cv::circle(img, projectedKeypoints[3], 3, cv::Scalar(255,255,0), -1);  // cyan logo_pin
	//cv::circle(img, projectedKeypoints[5], 3, cv::Scalar(0,255,255), -1);  // yellow logo_wheel
	//cv::circle(img, projectedKeypoints[7], 3, cv::Scalar(255,255,0), -1);  // cyan logo_s
	//cv::circle(img, projectedKeypoints[11], 3, cv::Scalar(255,255,0), -1);  // cyan logo_pivot


	// TEST cloud
	//    std::cout << cHpkey[7] << "; " << cHpkey[11] << std::endl;
	//    std::cout << cloud_.at<cv::Vec3f>(projectedKeypoints[7].y, projectedKeypoints[7].x)
	//            << cloud_.at<cv::Vec3f>(projectedKeypoints[11].y, projectedKeypoints[11].x)<< std::endl;
	//    std::cout << "==================================================================" << std::endl;
}

void CameraProcessor::draw_skel(cv::Mat &img, const cv::Mat &cHb,
								 const Eigen::Matrix<double, 4, 4, Eigen::RowMajor> &psm_Hj4,
								 const Eigen::Matrix<double, 4, 4, Eigen::RowMajor> &psm_Hj5,
								 const Eigen::Matrix<double, 4, 4, Eigen::RowMajor> &psm_He,
								 const float jaw_in, const cv::Mat& err_T, int psm_num, float slope)
{
	cv::Mat subP = cv::Mat::eye(3, 3, CV_32FC1);
	subP.at<float>(0,0) = Fx; subP.at<float>(1,1) = Fy;
	subP.at<float>(0,2) = Px; subP.at<float>(1,2) = Py;

	std::vector<cv::Mat> bHj_vec;
	std::vector<cv::Point3f> cPj(7); // 1 shaft_far, 3 joint, 3 tips (cen/l/r)

	for (int i = 0; i < 3; i++)
		bHj_vec.push_back(cv::Mat(4, 4, CV_64F));


	::memcpy(bHj_vec[0].data, psm_Hj4.data(), sizeof(double)*16);
	::memcpy(bHj_vec[1].data, psm_Hj5.data(), sizeof(double)*16);
	::memcpy(bHj_vec[2].data, psm_He.data(), sizeof(double)*16);

	cv::Mat corr_cHb = err_T * cHb;

	for (int i = 1; i <= bHj_vec.size(); i++)
	{
		// NB: bHj_vec unit in mm
		cv::Mat crHj_curr = corr_cHb * bHj_vec[i-1];
		cPj[i].x = crHj_curr.at<double>(0, 3);
		cPj[i].y = crHj_curr.at<double>(1, 3);
		cPj[i].z = abs(crHj_curr.at<double>(2, 3));    // abs to ensure projection make sense
	}

	// NB: Shaft line
	cv::Mat j2H_offset = cv::Mat::eye(4,4,CV_64FC1);
	j2H_offset.at<double>(0, 3) = 50;  // 200mm
	cv::Mat temp = corr_cHb * bHj_vec[0] * j2H_offset;
	cPj[0].x = temp.at<double>(0, 3);
	cPj[0].y = temp.at<double>(1, 3);
	cPj[0].z = abs(temp.at<double>(2, 3));    // abs to ensure projection make sense

	// Apply tip offset
	cv::Mat crHt;
	if (psm_num == 1)
		crHt = corr_cHb * bHj_vec.back() * simulator_->psm1_tool->Tip_Mat();
	else if (psm_num == 2)
		crHt = corr_cHb * bHj_vec.back() * simulator_->psm2_tool->Tip_Mat();

	cPj[4].x = crHt.at<double>(0, 3);
	cPj[4].y = crHt.at<double>(1, 3);
	cPj[4].z = crHt.at<double>(2, 3);

	/* Draw for extra points */
	std::vector<cv::Point3f> cHpkey = simulator_->calc_keypoints(bHj_vec[0], bHj_vec[1], bHj_vec[2], jaw_in, corr_cHb, psm_num);


	cPj[5] = cHpkey[12];
	cPj[6] = cHpkey[13];

	// cPj[0]: shaft_far end
	// cPj[1]: shaft
	// cPj[2]: logo
	// cPj[3]: end-effector
	// cPj[4]: tip central
	// cPj[5]: tip flat (right)
	// cPj[6]: tip deep (left)

	std::vector<cv::Point2f> projectedPoints;
	cv::Mat rVec, tVec;
	cv::Mat distCoeffs;
	rVec = cv::Mat::zeros(3,1,CV_32FC1); tVec = cv::Mat::zeros(3,1,CV_32FC1);
	cv::projectPoints(cPj, rVec, tVec, subP, distCoeffs, projectedPoints);


	//float slope = (projectedPoints[1].y - projectedPoints[0].y) / (projectedPoints[1].x - projectedPoints[0].x);
	cv::Point p(0,0), q(img.cols,img.rows);
	if (psm_num == 2)//projectedPoints[0].x < projectedPoints[1].x)
	{
		p.y = -(projectedPoints[1].x - p.x) * slope + projectedPoints[1].y;
		cv::line(img, projectedPoints[1], p, cv::Scalar(0,255,128), 2);  // shaft -> shaft_far
	}
	else
	{
		q.y = -(projectedPoints[1].x - q.x) * slope + projectedPoints[1].y;
		cv::line(img, projectedPoints[1], q, cv::Scalar(0,255,128), 2);  // shaft -> shaft_far
	}

	//cv::line(img, projectedPoints[1], projectedPoints[0], cv::Scalar(0,255,128), 2);  // shaft -> shaft_far
	cv::line(img, projectedPoints[3], projectedPoints[2], cv::Scalar(0,255,128), 2);  // end -> logo
	//cv::line(img, projectedPoints[4], projectedPoints[3], cv::Scalar(0,255,128), 2);  // tip -> jaw
	cv::line(img, projectedPoints[5], projectedPoints[3], cv::Scalar(0,255,128), 2);  // tip_flat -> jaw
	cv::line(img, projectedPoints[6], projectedPoints[3], cv::Scalar(0,255,128), 2);  // tip_deep -> jaw

	cv::circle(img, projectedPoints[2], 5, cv::Scalar(0,200,0), -1);   // logo
	cv::circle(img, projectedPoints[3], 5, cv::Scalar(0,200,0), -1);   // jaw
	//cv::circle(img, projectedPoints[4], 5, cv::Scalar(0,200,0), -1);    // tip central
	cv::circle(img, projectedPoints[5], 3, cv::Scalar(255,0,255), -1);    // tip left (deep)
	cv::circle(img, projectedPoints[6], 3, cv::Scalar(0,255,0), -1);    // tip right (flat)

	//cv::circle(img, projectedPoints[2], 10, cv::Scalar(0,255,0), -1);   // logo
	//cv::circle(img, projectedPoints[3], 10, cv::Scalar(0,0,255), -1);   // jaw
	//cv::circle(img, projectedPoints[4], 5, cv::Scalar(255,0,0), -1);    // tip central
	//cv::circle(img, projectedPoints[5], 3, cv::Scalar(0,255,0), -1);    // tip left (deep)
	//cv::circle(img, projectedPoints[6], 3, cv::Scalar(0,0,255), -1);    // tip right (flat)

	//cv::line(img, projectedPoints[1], projectedPoints[0], cv::Scalar(0,255,0), 2);  // shaft -> shaft_far
	//cv::line(img, projectedPoints[3], projectedPoints[2], cv::Scalar(0,0,255), 2);  // end -> logo
	//cv::line(img, projectedPoints[4], projectedPoints[3], cv::Scalar(255,0,0), 2);  // tip -> jaw
	//cv::line(img, projectedPoints[5], projectedPoints[3], cv::Scalar(255,0,0), 2);  // tip_flat -> jaw
	//cv::line(img, projectedPoints[6], projectedPoints[3], cv::Scalar(255,0,0), 2);  // tip_deep -> jaw

}



void decompose_rotation_xyz(const cv::Mat &R, double& thetaX, 
							double& thetaY, double& thetaZ)
{

	// R = Rx * Ry * Rz order. R is CV_64F.
	thetaX = atan2(-R.at<double>(1, 2), R.at<double>(2, 2));
	thetaY = atan2(R.at<double>(0, 2), sqrt(R.at<double>(1, 2) * R.at<double>(1, 2) 
		+ R.at<double>(2, 2) * R.at<double>(2, 2)));
	thetaZ = atan2(-R.at<double>(0, 1), R.at<double>(0, 0));
	// MATLAB:
	//x = atan2(-R(2,3), R(3,3));
	//y = atan2(R(1,3), sqrt(R(2,3)*R(2,3) + R(3,3)*R(3,3)));
	//z = atan2(-R(1,2), R(1,1));

}

void compose_rotation(const double &thetaX, const double &thetaY,
					  const double &thetaZ, cv::Mat &R)
{
	cv::Mat X = cv::Mat::eye(3, 3, CV_64F);
	cv::Mat Y = cv::Mat::eye(3, 3, CV_64F);
	cv::Mat Z = cv::Mat::eye(3, 3, CV_64F);

	X.at<double>(1, 1) = cos(thetaX);
	X.at<double>(1, 2) = -sin(thetaX);
	X.at<double>(2, 1) = sin(thetaX);
	X.at<double>(2, 2) = cos(thetaX);

	Y.at<double>(0, 0) = cos(thetaY);
	Y.at<double>(0, 2) = sin(thetaY);
	Y.at<double>(2, 0) = -sin(thetaY);
	Y.at<double>(2, 2) = cos(thetaY);

	Z.at<double>(0, 0) = cos(thetaZ);
	Z.at<double>(0, 1) = -sin(thetaZ);
	Z.at<double>(1, 0) = sin(thetaZ);
	Z.at<double>(1, 1) = cos(thetaZ);

	R = X * Y * Z; // R is CV_64F. 
}

QImage CameraProcessor::cvtCvMat2QImage(const cv::Mat & image)
{
	QImage qtemp;
	if(!image.empty() && image.depth() == CV_8U)
	{
		const unsigned char * data = image.data;
		qtemp = QImage(image.cols, image.rows, QImage::Format_RGB32);
		for(int y = 0; y < image.rows; ++y, data += image.cols*image.elemSize())
		{
			for(int x = 0; x < image.cols; ++x)
			{
				QRgb * p = ((QRgb*)qtemp.scanLine (y)) + x;
				*p = qRgb(data[x * image.channels()+2], data[x * image.channels()+1], data[x * image.channels()]);
			}
		}
	}
	else if(!image.empty() && image.depth() != CV_8U)
	{
		printf("Wrong image format, must be 8_bits\n");
	}
	return qtemp;
}


#ifdef __linux__
// Framerate calculation
double CameraProcessor::time_to_double(timeval *t)
{
	return (t->tv_sec + (t->tv_usec/1000000.0));
}

double CameraProcessor::time_diff(timeval *t1, timeval *t2)
{
	return time_to_double(t2) - time_to_double(t1);
}
#endif
