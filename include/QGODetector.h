/**
 * QGODetector.h
 * 
 * Reference:
 * M. Ye, et al. "Real-time 3D Tracking of Articulated Tools for Robotic Surgery". 
 * MICCAI 2016.
 *
 * @author  Menglong Ye, Imperial College London
 * @contact magnumye@gmail.com
 * @license BSD License
 */
#pragma once

#include "QuantisedGradOrientation.h"

class QGODetector
{
public:
	QGODetector();
	~QGODetector();

	void match(const cv::Mat& source, float threshold, std::vector<Match>& la_matches,
		const std::vector<std::string>& la_class_ids, std::vector<Match>& ra_matches,
		const std::vector<std::string>& ra_class_ids, cv::Mat& quantized_image,
		cv::Mat& magnitude, int img_id);

	void addDualTemplateSet(const cv::Mat& source, const std::vector<cv::Rect>& la_part_boxes,
		const std::vector<std::string>& la_class_names, const std::vector<cv::Rect>& ra_part_boxes,
		const std::vector<std::string>& ra_class_names, cv::Mat& quantized_angle,
		cv::Mat& magnitude, std::vector<cv::Rect>& la_bounding_boxes,
		std::vector<cv::Rect>& ra_bounding_boxes,
		std::vector<bool>& la_states, std::vector<bool>& ra_states);

	void colormap(const cv::Mat& quantized, cv::Mat& dst);

private:
	struct MatchPredicate
	{
	  MatchPredicate(float _threshold) : threshold(_threshold) {}
	  bool operator() (const Match& m) { return m.similarity < threshold; }
	  float threshold;
	};
	
	typedef std::vector<cv::Mat> LinearMemories;
	typedef std::map<std::string, Template> TemplateMap;
	typedef std::map<std::string, std::vector<Template>> TemplateMaps;

	TemplateMap la_class_templates;
	TemplateMap ra_class_templates;

	int T, ST;
	int gamma;
	QuantisedGradOrientation QGO;

	void matchClass(const LinearMemories& memories, float threshold, cv::Size size,
		std::vector<Match>& matches, const std::string& class_id, const Template& class_template);
	void matchClass(const LinearMemories& memories, float threshold, cv::Size size,
		std::vector<Match>& matches, const std::string& class_id, const Template& class_template,
		cv::Mat& similarity, int img_id, int psm_num);
};
