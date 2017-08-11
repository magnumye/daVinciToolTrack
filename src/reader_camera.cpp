#include "reader_camera.h"
#include <iostream>


ReaderCamera::ReaderCamera(std::string path, std::string stereo_calib_path) :
    frame_count (0)
  , total_num_frame (0)
{
    if (!vid_cap.open(path))
    {
        std::cout << "Error while reading camera..." << std::endl;
        exit(1);
    }

    m_stereo_ = new StereoBase(stereo_calib_path);

    total_num_frame = vid_cap.get(CV_CAP_PROP_FRAME_COUNT);
    video_data_L.resize(total_num_frame);
    video_data_R.resize(total_num_frame);
    for (int i = 0; i < total_num_frame; i++)
    {
        cv::Mat stereo_img;
        vid_cap >> stereo_img; // get a new frame from camera
        cv::Mat temp_left, temp_right, rectify_left, rectify_right;
        temp_left = cv::Mat(stereo_img, cv::Rect(0,0,stereo_img.cols/2,stereo_img.rows));
        temp_right = cv::Mat(stereo_img, cv::Rect(stereo_img.cols/2,0,stereo_img.cols/2,stereo_img.rows));
        m_stereo_->setInputImages(temp_left, temp_right);
        m_stereo_->getRectifiedImage(rectify_left, rectify_right);
        video_data_L[i] = rectify_left.clone();
        video_data_R[i] = rectify_right.clone();
    }

    vid_cap.release();
}

ReaderCamera::~ReaderCamera()
{
    if (m_stereo_ != NULL)
        delete m_stereo_;
}

void ReaderCamera::read_curr_image(cv::Mat &img_out)
{
    if (frame_count < total_num_frame)
    {
        video_data_L[frame_count].copyTo(img_out);
        frame_count++;
    }
    else
    {
        video_data_L[total_num_frame - 1].copyTo(img_out);
    }
}
