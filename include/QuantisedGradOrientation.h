/**
 * QuantisedGradOrientation.h
 * This soruce code was partially adapted from OPENCV, corresponding to the approach introduced in [Hinterstoisser et al. IEEE TPAMI 2017]. 
 * Reference:
 * M. Ye, et al. "Real-time 3D Tracking of Articulated Tools for Robotic Surgery". 
 * MICCAI 2016.
 *
 * @author  Menglong Ye, Imperial College London
 * @contact magnumye@gmail.com
 * @license BSD License
 */

#pragma once 

#include <opencv2/opencv.hpp>

#include "BasicStructs.h"


class QuantisedGradOrientation
{
public:
	QuantisedGradOrientation();
	~QuantisedGradOrientation();

	void hysteresisGradient(cv::Mat& magnitude, cv::Mat& quantized_angle,
		cv::Mat& angle, float threshold);
	void quantizedOrientations(const cv::Mat& src, cv::Mat& magnitude,
		cv::Mat& angle, float threshold);
	cv::Mat displayQuantized(const cv::Mat& quantized);
	inline int getLabel(int quantized)
	{
		switch (quantized)
		{
		case 1:   return 0;
		case 2:   return 1;
		case 4:   return 2;
		case 8:   return 3;
		case 16:  return 4;
		case 32:  return 5;
		case 64:  return 6;
		case 128: return 7;
		default:
			CV_Error(CV_StsBadArg, "Invalid value of quantized parameter");
			return -1; //avoid warning
		}
	}
	void colormap(const cv::Mat& quantized, cv::Mat& dst);


	bool extractTemplate(Template& templ, float strong_threshold, 
		size_t num_features, const cv::Mat& magnitude, 
		const cv::Mat& angle, const cv::Mat& mask, cv::Rect& box);
	
	bool extractTemplateNoResize(Template& templ, float strong_threshold, 
		size_t num_features, const cv::Mat& magnitude, 
		const cv::Mat& angle, const cv::Mat& mask, cv::Rect& box);

	bool extractTemplateWithResize(Template& templ, float strong_threshold, 
		size_t num_features, const cv::Mat& magnitude, 
		const cv::Mat& angle, const cv::Mat& mask, cv::Rect& box, int ratio);
	/**
	* \brief Choose candidate features so that they are not bunched together.
	*
	* \param[in]  candidates   Candidate features sorted by score.
	* \param[out] features     Destination vector of selected features.
	* \param[in]  num_features Number of candidates to select.
	* \param[in]  distance     Hint for desired distance between features.
	*/
	void selectScatteredFeatures(const std::vector<Candidate>& candidates,
		std::vector<Feature>& features,
		size_t num_features, float distance);

	///////// Response maps /////////////
	void orUnaligned8u(const uchar * src, const int src_stride,
                   uchar * dst, const int dst_stride,
                   const int width, const int height);
	void spread(const cv::Mat& src, cv::Mat& dst, int T);
	void computeResponseMaps(const cv::Mat& src, std::vector<cv::Mat>& response_maps);
	void linearize(const cv::Mat& response_map, cv::Mat& linearized, int T);

	///////// Linearized similarities /////////////
	const unsigned char* accessLinearMemory(const std::vector<cv::Mat>& linear_memories,
          const Feature& f, int T, int W);
	void similarity(const std::vector<cv::Mat>& linear_memories, const Template& templ,
                cv::Mat& dst, cv::Size size, int T);
	void similarityLocal(const std::vector<cv::Mat>& linear_memories, const Template& templ,
                     cv::Mat& dst, cv::Size size, int T, cv::Point center);
	void addUnaligned8u16u(const uchar * src1, const uchar * src2, ushort * res, int length);
	void addSimilarities(const std::vector<cv::Mat>& similarities, cv::Mat& dst);

private:
	float weakThreshold; // 10.0
	size_t numFeatures; //63
	float strongThreshold; //55
};
