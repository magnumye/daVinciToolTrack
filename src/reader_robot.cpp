#include "reader_robot.h"


ReaderRobot::ReaderRobot(std::string path) :
    frame_count (0)
{
    fs.open(path, cv::FileStorage::READ);
    fs["J4_poses"] >> J4_poses;
    fs["J5_poses"] >> J5_poses;
    fs["bHe_poses"] >> bHe_poses;
    fs["jaws"] >> jaws;

}

void ReaderRobot::read_pose(Eigen::Matrix<double, 4, 4, Eigen::RowMajor> &J4_pose,
                            Eigen::Matrix<double, 4, 4, Eigen::RowMajor> &J5_pose,
                            Eigen::Matrix<double, 4, 4, Eigen::RowMajor> &bHe_pose,
                            float &jaw
                            )
{
    ::memcpy(J4_pose.data(), J4_poses[frame_count].data, 16 * sizeof(double));
    ::memcpy(J5_pose.data(), J5_poses[frame_count].data, 16 * sizeof(double));
    ::memcpy(bHe_pose.data(), bHe_poses[frame_count].data, 16 * sizeof(double));

    jaw = jaws[frame_count];

    frame_count++;
    if (frame_count >= bHe_poses.size())
        exit(0); // not a good idea
}
