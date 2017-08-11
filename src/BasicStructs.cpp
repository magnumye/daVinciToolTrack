#include "BasicStructs.h"

void Feature::read(const cv::FileNode& fn)
{
  cv::FileNodeIterator fni = fn.begin();
  fni >> x >> y >> label;
}

void Feature::write(cv::FileStorage& fs) const
{
  fs << "[:" << x << y << label << "]";
}

void Template::read(const cv::FileNode& fn)
{
  width = fn["width"];
  height = fn["height"];
  pyramid_level = fn["pyramid_level"];

  cv::FileNode features_fn = fn["features"];
  features.resize(features_fn.size());
  cv::FileNodeIterator it = features_fn.begin(), it_end = features_fn.end();
  for (int i = 0; it != it_end; ++it, ++i)
  {
    features[i].read(*it);
  }
}

void Template::write(cv::FileStorage& fs) const
{
  fs << "width" << width;
  fs << "height" << height;
  fs << "pyramid_level" << pyramid_level;

  fs << "features" << "[";
  for (int i = 0; i < (int)features.size(); ++i)
  {
    features[i].write(fs);
  }
  fs << "]"; // features
}

