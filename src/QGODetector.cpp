#include "QGODetector.h"
#include <fstream>

//Uncomment below to speeding up
//#define RATIO 1.5


QGODetector::QGODetector()
{
	T = 4;
	ST = 5;
	//Gradient magnitude threshold
	gamma = 50;
}

QGODetector::~QGODetector()
{
}

void QGODetector::match(const cv::Mat& source, float threshold, std::vector<Match>& la_matches,
						const std::vector<std::string>& la_class_ids, std::vector<Match>& ra_matches,
						const std::vector<std::string>& ra_class_ids, cv::Mat& quantized_image,
						cv::Mat& magnitude, int img_id)
{
	la_matches.clear();
	ra_matches.clear();	

#ifdef RATIO
	LinearMemories memories(8);

	cv::Mat quantized, spread_quantized;
	std::vector<cv::Mat> response_maps;

	float ratio = RATIO;
	cv::Mat resized_source;
	cv::resize(source, resized_source, cv::Size(source.cols/ratio, source.rows/ratio));
	// check if dividable by T
	int w = resized_source.cols; int h = resized_source.rows;
	w = w - w%16; h = h - h%16;
	cv::Mat source_cropped = resized_source(cv::Rect(0, 0, w, h));

	QGO.quantizedOrientations(source_cropped, magnitude, quantized, gamma-30);
	QGO.spread(quantized, spread_quantized, ST);
	QGO.computeResponseMaps(spread_quantized, response_maps);

	for (int i = 0; i < 8; i++)
	{
		QGO.linearize(response_maps[i], memories[i], T);
	}

	quantized.copyTo(quantized_image);

	cv::Size sz = quantized.size();

	if (la_class_ids.empty())
	{
		// Match all templates
		TemplateMap::const_iterator it = la_class_templates.begin(), itend = la_class_templates.end();
		for ( ; it != itend; ++it)
		{
			cv::Mat sim;
			matchClass(memories, threshold, sz, la_matches, it->first, it->second, sim, img_id, 2);

		}
	}
	else
	{
		// Match only templates for the requested class IDs
		for (int i = 0; i < (int)la_class_ids.size(); ++i)
		{
			TemplateMap::const_iterator it = la_class_templates.find(la_class_ids[i]);
			if (it != la_class_templates.end())
			{
				cv::Mat sim;
				matchClass(memories, threshold, sz, la_matches, it->first, it->second, sim, img_id, 2);

			}
		}
	}

	if (ra_class_ids.empty())
	{
		// Match all templates
		TemplateMap::const_iterator it = ra_class_templates.begin(), itend = ra_class_templates.end();
		for ( ; it != itend; ++it)
		{
			cv::Mat sim;
			matchClass(memories, threshold, sz, ra_matches, it->first, it->second, sim, img_id, 1);

		}
	}
	else
	{
		// Match only templates for the requested class IDs
		for (int i = 0; i < (int)ra_class_ids.size(); ++i)
		{
			TemplateMap::const_iterator it = ra_class_templates.find(ra_class_ids[i]);
			if (it != ra_class_templates.end())
			{
				cv::Mat sim;
				matchClass(memories, threshold, sz, ra_matches, it->first, it->second, sim, img_id, 1);

			}
		}
	}

	// Sort matches by similarity
	std::sort(la_matches.begin(), la_matches.end());
	std::vector<Match>::iterator la_new_end = std::unique(la_matches.begin(), la_matches.end());
	la_matches.erase(la_new_end, la_matches.end());

	std::sort(ra_matches.begin(), ra_matches.end());
	std::vector<Match>::iterator ra_new_end = std::unique(ra_matches.begin(), ra_matches.end());
	ra_matches.erase(ra_new_end, ra_matches.end());

	for (int i = 0; i < la_matches.size(); i++)
	{
		la_matches[i].x *= ratio;
		la_matches[i].y *= ratio;
	}
	for (int i = 0; i < ra_matches.size(); i++)
	{
		ra_matches[i].x *= ratio;
		ra_matches[i].y *= ratio;
	}
#else
	LinearMemories memories(8);

	cv::Mat quantized, spread_quantized;
	std::vector<cv::Mat> response_maps;

	// check if dividable by T
	int w = source.cols; int h = source.rows;
	w = w - w%T; h = h - h%T;
	cv::Mat source_cropped = source(cv::Rect(0, 0, w, h));

	QGO.quantizedOrientations(source_cropped, magnitude, quantized, gamma);
	QGO.spread(quantized, spread_quantized, ST);
	QGO.computeResponseMaps(spread_quantized, response_maps);

	for (int i = 0; i < 8; i++)
	{
		QGO.linearize(response_maps[i], memories[i], T);
	}

	quantized.copyTo(quantized_image);

	cv::Size sz = quantized.size();

	if (la_class_ids.empty())
	{
		// Match all templates
		TemplateMap::const_iterator it = la_class_templates.begin(), itend = la_class_templates.end();
		for ( ; it != itend; ++it)
		{
			cv::Mat sim;
			matchClass(memories, threshold, sz, la_matches, it->first, it->second, sim, img_id, 2);
			//std::cout << it->first << std::endl;
		}
		//std::cout << std::endl;
	}
	else
	{
		// Match only templates for the requested class IDs
		for (int i = 0; i < (int)la_class_ids.size(); ++i)
		{
			TemplateMap::const_iterator it = la_class_templates.find(la_class_ids[i]);
			if (it != la_class_templates.end())
			{
				cv::Mat sim;
				matchClass(memories, threshold, sz, la_matches, it->first, it->second, sim, img_id, 2);

			}
		}
	}

	if (ra_class_ids.empty())
	{
		// Match all templates
		TemplateMap::const_iterator it = ra_class_templates.begin(), itend = ra_class_templates.end();
		for ( ; it != itend; ++it)
		{
			cv::Mat sim;
			matchClass(memories, threshold, sz, ra_matches, it->first, it->second, sim, img_id, 1);
			//std::cout << it->first << std::endl;
		}
		//std::cout << std::endl;
	}
	else
	{
		// Match only templates for the requested class IDs
		for (int i = 0; i < (int)ra_class_ids.size(); ++i)
		{
			TemplateMap::const_iterator it = ra_class_templates.find(ra_class_ids[i]);
			if (it != ra_class_templates.end())
			{
				cv::Mat sim;
				matchClass(memories, threshold, sz, ra_matches, it->first, it->second, sim, img_id, 1);

			}
		}
	}

	// Sort matches by similarity
	std::sort(la_matches.begin(), la_matches.end());
	std::vector<Match>::iterator la_new_end = std::unique(la_matches.begin(), la_matches.end());
	la_matches.erase(la_new_end, la_matches.end());

	std::sort(ra_matches.begin(), ra_matches.end());
	std::vector<Match>::iterator ra_new_end = std::unique(ra_matches.begin(), ra_matches.end());
	ra_matches.erase(ra_new_end, ra_matches.end());

#endif

}


void QGODetector::matchClass(const LinearMemories& memories, float threshold, cv::Size sz,
							 std::vector<Match>& matches, const std::string& class_id, 
							 const Template& class_template)
{
	cv::Mat sim;
	int num_features = 0;

	const Template& templ = class_template;
	num_features += static_cast<int>(templ.features.size());
	QGO.similarity(memories, templ, sim, sz, T);
	//cv::imshow("Sim1", sim);
	// Calc overall similarity
	cv::Mat total_similarity;


	//std::vector<cv::Mat> similarities;
	//similarities.push_back(sim);
	//QGO.addSimilarities(similarities, total_similarity);

	// There is only one, so convert it immediately.
	sim.convertTo(total_similarity, CV_16U);

	// Convert user-friendly percentage to raw similarity threshold. The percentage
	// threshold scales from half the max response (what you would expect from applying
	// the template to a completely random image) to the max response.
	// NOTE: This assumes max per-feature response is 4, so we scale between [2*nf, 4*nf].
	//int raw_threshold = static_cast<int>(2*num_features + (threshold / 100.f) * (2*num_features) + 0.5f);
	int raw_threshold = static_cast<int>(2*num_features + ((threshold - 50) / 100.f) * (2*num_features) + 0.5f);
	// Find initial matches
	std::vector<Match> candidates;
	int lowest_T = T; int template_id = 0; // There is only one online template
	for (int r = 0; r < total_similarity.rows; ++r)
	{
		ushort* row = total_similarity.ptr<ushort>(r);
		for (int c = 0; c < total_similarity.cols; ++c)
		{
			int raw_score = row[c];
			if (raw_score > raw_threshold)
			{
				int offset = lowest_T / 2 + (lowest_T % 2 - 1);
				int x = c * lowest_T;// + offset;
				int y = r * lowest_T;// + offset;
				float score =(raw_score * 100.f) / (4 * num_features) + 0.5f;
				candidates.push_back(Match(x, y, score, class_id, static_cast<int>(template_id)));
			}
		}
	}

	// Locally refine each match
	int border = 8 * T;
	int offset = T / 2 + (T % 2 - 1);
	int max_x = sz.width - templ.width - border;
	int max_y = sz.height - templ.height - border;

	cv::Mat sim2, total_similarity2;
	for (int m = 0; m < (int)candidates.size(); ++m)
	{
		Match& match2 = candidates[m];
		int x = match2.x;//match2.x * 2 + 1; /// @todo Support other pyramid distance
		int y = match2.y;//match2.y * 2 + 1;

		// Require 8 (reduced) row/cols to the up/left
		x = std::max(x, border);
		y = std::max(y, border);

		// Require 8 (reduced) row/cols to the down/left, plus the template size
		x = std::min(x, max_x);
		y = std::min(y, max_y);

		// Compute local similarity maps for each modality
		int numFeatures = 0;

		numFeatures += static_cast<int>(templ.features.size());
		QGO.similarityLocal(memories, templ, sim2, sz, T, cv::Point(x, y));

		sim2.convertTo(total_similarity2, CV_16U);

		// Find best local adjustment
		int best_score = 0;
		int best_r = -1, best_c = -1;
		for (int r = 0; r < total_similarity2.rows; ++r)
		{
			ushort* row = total_similarity2.ptr<ushort>(r);
			for (int c = 0; c < total_similarity2.cols; ++c)
			{
				int score = row[c];
				if (score > best_score)
				{
					best_score = score;
					best_r = r;
					best_c = c;
				}
			}
		}
		// Update current match
		match2.x = (x / T - 8 + best_c) * T; // + offset;
		match2.y = (y / T - 8 + best_r) * T; // + offset;
		match2.similarity = (best_score * 100.f) / (4 * numFeatures);
	}
	// Filter out any matches that drop below the similarity threshold
	std::vector<Match>::iterator new_end = std::remove_if(candidates.begin(), candidates.end(),
		MatchPredicate(threshold));
	candidates.erase(new_end, candidates.end());

	matches.insert(matches.end(), candidates.begin(), candidates.end());
}

void QGODetector::matchClass(const LinearMemories& memories, float threshold, cv::Size sz,
							 std::vector<Match>& matches, const std::string& class_id, 
							 const Template& class_template, cv::Mat& sim, 
							 int img_id, int psm_num)
{


	//cv::Mat sim;
	int num_features = 0;

	const Template& templ = class_template;
	num_features += static_cast<int>(templ.features.size());
	QGO.similarity(memories, templ, sim, sz, T);
	//cv::imshow("Sim1", sim);
	// Calc overall similarity
	cv::Mat total_similarity;


	//std::vector<cv::Mat> similarities;
	//similarities.push_back(sim);
	//QGO.addSimilarities(similarities, total_similarity);

	// There is only one, so convert it immediately.
	sim.convertTo(total_similarity, CV_16U);

	// Convert user-friendly percentage to raw similarity threshold. The percentage
	// threshold scales from half the max response (what you would expect from applying
	// the template to a completely random image) to the max response.
	// NOTE: This assumes max per-feature response is 4, so we scale between [2*nf, 4*nf].
	//int raw_threshold = static_cast<int>(2*num_features + (threshold / 100.f) * (2*num_features) + 0.5f);
	int raw_threshold = static_cast<int>(2*num_features + ((threshold - 50) / 100.f) * (2*num_features) + 0.5f);
	// Find initial matches
	std::vector<Match> candidates;
	//candidates.reserve(total_similarity.rows*total_similarity.cols);
	unsigned int candidate_num = 0;
	int lowest_T = T; int template_id = 0; // There is only one online template
	for (int r = 0; r < total_similarity.rows; ++r)
	{
		ushort* row = total_similarity.ptr<ushort>(r);
		for (int c = 0; c < total_similarity.cols; ++c)
		{
			int raw_score = row[c];
			if (raw_score > raw_threshold)
			{
				candidate_num++;
			}
		}
	}
	candidates.resize(candidate_num);
	unsigned int candidate_id = 0;;
	for (int r = 0; r < total_similarity.rows; ++r)
	{
		ushort* row = total_similarity.ptr<ushort>(r);
		for (int c = 0; c < total_similarity.cols; ++c)
		{
			int raw_score = row[c];
			if (raw_score > raw_threshold)
			{
				//int offset = lowest_T / 2 + (lowest_T % 2 - 1);
				int x = c * lowest_T;// + offset;
				int y = r * lowest_T;// + offset;
				float score =(raw_score * 100.f) / (4 * num_features) + 0.5f;
				candidates[candidate_id].x = x;
				candidates[candidate_id].y = y;
				candidates[candidate_id].similarity = score;
				//candidates[candidate_id].class_id = class_id;
				candidates[candidate_id].template_id = template_id;
				candidate_id++;
				//candidates.push_back(Match(x, y, score, class_id, static_cast<int>(template_id)));

			}

		}
	}
	assert(candidate_id == candidate_num);
	//std::vector<Match>::iterator new_end = std::unique(candidates.begin(), candidates.end());
	//candidates.erase(new_end, candidates.end());

	// Sort out nTop candidates, and keep them
	int maxMatchesPerPart = 10;
	int nTop = std::min(maxMatchesPerPart, (int)candidates.size());
	//std::cout<< candidates.size()<<std::endl;
	std::partial_sort(candidates.begin(), candidates.begin()+nTop, candidates.end());

	for (int i = 0; i < nTop; i++)
	{
		candidates[i].class_id = class_id;
	}

	matches.insert(matches.end(), candidates.begin(), candidates.begin()+nTop);
}




void QGODetector::addDualTemplateSet(const cv::Mat& source, const std::vector<cv::Rect>& la_part_boxes,
									 const std::vector<std::string>& la_class_names, const std::vector<cv::Rect>& ra_part_boxes,
									 const std::vector<std::string>& ra_class_names, cv::Mat& quantized_angle,
									 cv::Mat& magnitude, std::vector<cv::Rect>& la_bounding_boxes,
									 std::vector<cv::Rect>& ra_bounding_boxes,
									 std::vector<bool>& la_states, std::vector<bool>& ra_states)
{
	// clear previous templates
	la_class_templates.clear();
	ra_class_templates.clear();

	la_states.resize(la_part_boxes.size(), false);
	ra_states.resize(ra_part_boxes.size(), false);
	la_bounding_boxes.resize(la_part_boxes.size());
	ra_bounding_boxes.resize(ra_part_boxes.size());

	cv::Mat quantized;
#ifdef RATIO
	{
		float ratio = RATIO;
		cv::Mat resized_source;
		cv::resize(source, resized_source, cv::Size(source.cols/ratio, source.rows/ratio));
		QGO.quantizedOrientations(resized_source, magnitude, quantized, gamma+20);

		for (int i = 0; i < la_part_boxes.size(); i++)
		{
			Template tmpl;
			cv::Mat mask; // this mask is not used
			cv::Rect bb;
			cv::Rect roibb = la_part_boxes[i];
			roibb.x /= ratio;
			roibb.y /= ratio;
			roibb.width /= ratio;
			roibb.height /= ratio;
			la_states[i] = QGO.extractTemplateNoResize(tmpl, gamma+20, 63, magnitude(roibb), 
				quantized(roibb), mask, bb);

			// Translate into global image coordinates
			bb.x += roibb.x;
			bb.y += roibb.y;
			la_bounding_boxes[i] = bb;

			la_class_templates[la_class_names[i]] = tmpl;
		}

		for (int i = 0; i < ra_part_boxes.size(); i++)
		{
			Template tmpl;
			cv::Mat mask; // this mask is not used
			cv::Rect bb;
			cv::Rect roibb = ra_part_boxes[i];
			roibb.x /= ratio;
			roibb.y /= ratio;
			roibb.width /= ratio;
			roibb.height /= ratio;
			ra_states[i] = QGO.extractTemplateNoResize(tmpl, gamma+20, 63, magnitude(roibb), 
				quantized(roibb), mask, bb);

			// Translate into global image coordinates
			bb.x += roibb.x;
			bb.y += roibb.y;
			ra_bounding_boxes[i] = bb;

			ra_class_templates[ra_class_names[i]] = tmpl;
		}

		quantized.copyTo(quantized_angle);
	}
#else
	{
		QGO.quantizedOrientations(source, magnitude, quantized, gamma);

		for (int i = 0; i < la_part_boxes.size(); i++)
		{
			Template tmpl;
			cv::Mat mask; // this mask is not used
			cv::Rect bb;

			la_states[i] = QGO.extractTemplateNoResize(tmpl, gamma, 63, magnitude(la_part_boxes[i]), 
				quantized(la_part_boxes[i]), mask, bb);

			// Translate into global image coordinates
			bb.x += la_part_boxes[i].x;
			bb.y += la_part_boxes[i].y;
			la_bounding_boxes[i] = bb;

			la_class_templates[la_class_names[i]] = tmpl;
		}

		for (int i = 0; i < ra_part_boxes.size(); i++)
		{
			Template tmpl;
			cv::Mat mask; // this mask is not used
			cv::Rect bb;

			ra_states[i] = QGO.extractTemplateNoResize(tmpl, gamma, 63, magnitude(ra_part_boxes[i]), 
				quantized(ra_part_boxes[i]), mask, bb);

			// Translate into global image coordinates
			bb.x += ra_part_boxes[i].x;
			bb.y += ra_part_boxes[i].y;
			ra_bounding_boxes[i] = bb;

			ra_class_templates[ra_class_names[i]] = tmpl;
		}

		quantized.copyTo(quantized_angle);
	}
#endif
}

void QGODetector::colormap(const cv::Mat& quantized, cv::Mat& dst)
{
	QGO.colormap(quantized, dst);
}




