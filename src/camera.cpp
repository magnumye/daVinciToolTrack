#include "camera.h"


MonoCamera::MonoCamera(const std::string &calibration_filename)
{
    cv::FileStorage fs;

    try{

        cv::Mat cam_matrix;
        fs.open(calibration_filename,cv::FileStorage::READ);
        fs["P1"] >> cam_matrix;

        fx_ = (float)cam_matrix.at<double>(0, 0);
        fy_ = (float)cam_matrix.at<double>(1, 1);
        px_ = (float)cam_matrix.at<double>(0, 2);
        py_ = (float)cam_matrix.at<double>(1, 2);

        fs["D1"] >> distortion_params_;

        fs["image_Width"] >> image_width_;
        fs["image_Height"] >> image_height_;

        window_width_ = image_width_;
        window_height_ = image_height_;

    }catch(cv::Exception& e){

        std::cerr << "Error while reading from camara calibration file.\n" << e.msg << "\n";
        exit(1);

    }

    m_vtkcam_ = vtkSmartPointer<vtkCamera>::New();
}

MonoCamera::MonoCamera(const cv::Mat &intrinsic, const cv::Mat &distortion, const int image_width, const int image_height) :
    distortion_params_(distortion)
  , image_width_(image_width)
  , image_height_(image_height)
  , window_width_(image_width)
  , window_height_(image_height)
{

    fx_ = (float)intrinsic.at<double>(0, 0);
    fy_ = (float)intrinsic.at<double>(1, 1);
    px_ = (float)intrinsic.at<double>(0, 2);
    py_ = (float)intrinsic.at<double>(1, 2);

}

vtkSmartPointer<vtkCamera> MonoCamera::SetupCameraForRender()
{
    /**
     * View angle
     * */
    double fy_vtk = fy_;
//     if (window_height_ != image_height_)
//     {
//         double factor = static_cast<double>(window_height_)/static_cast<double>(image_height_);
//         fy_vtk = fy_ * factor;
//     }

    double view_angle = 2 * atan( ( window_height_ / 2 ) / fy_vtk ) * 180 / M_PI;

    m_vtkcam_->SetViewAngle(view_angle);

    /**
     * Window Center
     * */
    px_win_ = 0;
    double width = 0;

    py_win_ = 0;
    double height = 0;

    if( image_width_ != window_width_ || image_height_ != window_height_ )
    {

		double diff_X = (window_width_ - image_width_) / 2.0;
		double diff_Y = (window_height_ - image_height_) / 2.0;
		px_win_ = px_ + diff_X;
		py_win_ = py_ + diff_Y;

		width = window_width_;
		height = window_height_;

		std::cout << "origin principle: " << px_ << ", " << py_ << std::endl;
		std::cout << "modified principle: " << px_win_ << ", " << py_win_ << std::endl;

		std::cout << "origin size: " << image_width_ << ", " << image_height_ << std::endl;
		std::cout << "modified size: " << width << ", " << height << std::endl;

// 		std::cout << "Window size:" << width << ", " << height << std::endl;
    }
    else
    {
        px_win_ = px_;
        width = image_width_;

        py_win_ = py_;
        height = image_height_;
    }

    double cx = width - px_win_;
    double cy = py_win_;

    window_center.x = cx / ( ( width-1)/2 ) - 1 ;
    window_center.y = cy / ( ( height-1)/2 ) - 1;

    m_vtkcam_->SetWindowCenter(window_center.x, window_center.y);

    /**
     * Apply extrinsic
     * */
    cv::Mat m_scaled_mat = cv::Mat::eye(4, 4, CV_64FC1);
    m_scaled_mat.at<double>(1, 1) = -m_scaled_mat.at<double>(1, 1);
    m_scaled_mat.at<double>(2, 2) = -m_scaled_mat.at<double>(2, 2);

    cv::Mat m_scaled_transform = cv::Mat::eye(4, 4, CV_64FC1);

    m_scaled_transform = m_scaled_mat * wHc_.inv();

    cv::Mat m_rotation = m_scaled_transform(cv::Rect(0,0,3,3));

    // Normalise row of the rotation matrix
    for (unsigned int i = 0; i < 3; i++)    // for each row in the rotation
    {
        double norm_temp = 0.0;
        for (unsigned int j = 0; j < 3; j++)    // for each element in row
            norm_temp += m_rotation.at<double>(i, j) * m_rotation.at<double>(i, j);

        if (!isEqual(norm_temp, 0.0))
        {
            double scale = 1.0 / sqrt(norm_temp);
            for (unsigned int j = 0; j < 3; j++)    // for each element in row
                m_rotation.at<double>(i, j) = m_rotation.at<double>(i, j) * scale;
        }
    }

    cv::Mat m_rotation_inv = m_rotation.inv();
    cv::Mat m_translation = m_scaled_transform(cv::Rect(3, 0, 1, 3));

    // rotate translation vector by inverse rotation P = P'
    m_translation = m_rotation_inv * m_translation;
    m_translation *= -1; // save -P'

    // from here proceed as normal
    // focalPoint = P-viewPlaneNormal, viewPlaneNormal is rotation[2]
    cv::Vec3d m_view_plane_normal;
    m_view_plane_normal[0] = m_rotation.at<double>(2, 0);
    m_view_plane_normal[1] = m_rotation.at<double>(2, 1);
    m_view_plane_normal[2] = m_rotation.at<double>(2, 2);

    m_vtkcam_->SetPosition(m_translation.at<double>(0), m_translation.at<double>(1), m_translation.at<double>(2));

    m_vtkcam_->SetFocalPoint(m_translation.at<double>(0) - m_view_plane_normal[0],
                           m_translation.at<double>(1) - m_view_plane_normal[1],
                           m_translation.at<double>(2) - m_view_plane_normal[2]);

    m_vtkcam_->SetViewUp(m_rotation.at<double>(1,0), m_rotation.at<double>(1,1), m_rotation.at<double>(1,2));
    m_vtkcam_->SetClippingRange(20, 200);

    return m_vtkcam_;
}

void MonoCamera::set_window_size(int width, int height)
{
	window_width_ = width;
	window_height_ = height;
}

void MonoCamera::set_wHc(cv::Mat &H)
{
    H.copyTo(wHc_);

    /**
     * Apply extrinsic
     * */
    cv::Mat m_scaled_mat = cv::Mat::eye(4, 4, CV_64FC1);
    m_scaled_mat.at<double>(1, 1) = -m_scaled_mat.at<double>(1, 1);
    m_scaled_mat.at<double>(2, 2) = -m_scaled_mat.at<double>(2, 2);

    cv::Mat m_scaled_transform = cv::Mat::eye(4, 4, CV_64FC1);

    m_scaled_transform = m_scaled_mat * wHc_.inv();

    cv::Mat m_rotation = m_scaled_transform(cv::Rect(0,0,3,3));

    // Normalise row of the rotation matrix
    for (unsigned int i = 0; i < 3; i++)    // for each row in the rotation
    {
        double norm_temp = 0.0;
        for (unsigned int j = 0; j < 3; j++)    // for each element in row
            norm_temp += m_rotation.at<double>(i, j) * m_rotation.at<double>(i, j);

        if (!isEqual(norm_temp, 0.0))
        {
            double scale = 1.0 / sqrt(norm_temp);
            for (unsigned int j = 0; j < 3; j++)    // for each element in row
                m_rotation.at<double>(i, j) = m_rotation.at<double>(i, j) * scale;
        }
    }

    cv::Mat m_rotation_inv = m_rotation.inv();
    cv::Mat m_translation = m_scaled_transform(cv::Rect(3, 0, 1, 3));

    // rotate translation vector by inverse rotation P = P'
    m_translation = m_rotation_inv * m_translation;
    m_translation *= -1; // save -P'

    // from here proceed as normal
    // focalPoint = P-viewPlaneNormal, viewPlaneNormal is rotation[2]
    cv::Vec3d m_view_plane_normal;
    m_view_plane_normal[0] = m_rotation.at<double>(2, 0);
    m_view_plane_normal[1] = m_rotation.at<double>(2, 1);
    m_view_plane_normal[2] = m_rotation.at<double>(2, 2);

    m_vtkcam_->SetPosition(m_translation.at<double>(0), m_translation.at<double>(1), m_translation.at<double>(2));

    m_vtkcam_->SetFocalPoint(m_translation.at<double>(0) - m_view_plane_normal[0],
                           m_translation.at<double>(1) - m_view_plane_normal[1],
                           m_translation.at<double>(2) - m_view_plane_normal[2]);

    m_vtkcam_->SetViewUp(m_rotation.at<double>(1,0), m_rotation.at<double>(1,1), m_rotation.at<double>(1,2));
}

void MonoCamera::set_w2Hc(cv::Mat &H)
{
    H.copyTo(w2Hc_);
}

void MonoCamera::set_wHc_corr(cv::Mat &H)
{
    H.copyTo(wHc_corr_);

	
    /**
     * Apply extrinsic
     * */
    cv::Mat m_scaled_mat = cv::Mat::eye(4, 4, CV_64FC1);
    m_scaled_mat.at<double>(1, 1) = -m_scaled_mat.at<double>(1, 1);
    m_scaled_mat.at<double>(2, 2) = -m_scaled_mat.at<double>(2, 2);

    cv::Mat m_scaled_transform = cv::Mat::eye(4, 4, CV_64FC1);

    m_scaled_transform = m_scaled_mat * wHc_corr_.inv();

    cv::Mat m_rotation = m_scaled_transform(cv::Rect(0,0,3,3));

    // Normalise row of the rotation matrix
    for (unsigned int i = 0; i < 3; i++)    // for each row in the rotation
    {
        double norm_temp = 0.0;
        for (unsigned int j = 0; j < 3; j++)    // for each element in row
            norm_temp += m_rotation.at<double>(i, j) * m_rotation.at<double>(i, j);

        if (!isEqual(norm_temp, 0.0))
        {
            double scale = 1.0 / sqrt(norm_temp);
            for (unsigned int j = 0; j < 3; j++)    // for each element in row
                m_rotation.at<double>(i, j) = m_rotation.at<double>(i, j) * scale;
        }
    }

    cv::Mat m_rotation_inv = m_rotation.inv();
    cv::Mat m_translation = m_scaled_transform(cv::Rect(3, 0, 1, 3));

    // rotate translation vector by inverse rotation P = P'
    m_translation = m_rotation_inv * m_translation;
    m_translation *= -1; // save -P'

    // from here proceed as normal
    // focalPoint = P-viewPlaneNormal, viewPlaneNormal is rotation[2]
    cv::Vec3d m_view_plane_normal;
    m_view_plane_normal[0] = m_rotation.at<double>(2, 0);
    m_view_plane_normal[1] = m_rotation.at<double>(2, 1);
    m_view_plane_normal[2] = m_rotation.at<double>(2, 2);

    m_vtkcam_->SetPosition(m_translation.at<double>(0), m_translation.at<double>(1), m_translation.at<double>(2));

    m_vtkcam_->SetFocalPoint(m_translation.at<double>(0) - m_view_plane_normal[0],
                           m_translation.at<double>(1) - m_view_plane_normal[1],
                           m_translation.at<double>(2) - m_view_plane_normal[2]);

    m_vtkcam_->SetViewUp(m_rotation.at<double>(1,0), m_rotation.at<double>(1,1), m_rotation.at<double>(1,2));

}

void MonoCamera::set_w2Hc_corr(cv::Mat &H)
{
    H.copyTo(w2Hc_corr_);

}
