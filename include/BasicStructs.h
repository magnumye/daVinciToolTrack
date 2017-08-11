#pragma once

#include <opencv2/opencv.hpp>

// Adapted from OPENCV
/**
* \brief Discriminant feature described by its location and label.
*/
struct Feature
{
	int x; ///< x offset
	int y; ///< y offset
	int label; ///< Quantization

	Feature() : x(0), y(0), label(0) {}
	Feature(int x, int y, int label);

	void read(const cv::FileNode& fn);
	void write(cv::FileStorage& fs) const;
};

inline Feature::Feature(int _x, int _y, int _label) : x(_x), y(_y), label(_label) {}

struct Template
{
	int width;
	int height;
	int pyramid_level;
	std::vector<Feature> features;

	void read(const cv::FileNode& fn);
	void write(cv::FileStorage& fs) const;
};

/// Candidate feature with a score
struct Candidate
{
	Candidate(int x, int y, int label, float score);

	/// Sort candidates with high score to the front
	bool operator<(const Candidate& rhs) const
	{
		return score > rhs.score;
	}

	Feature f;
	float score;
};

inline Candidate::Candidate(int x, int y, int label, float _score) : f(x, y, label), score(_score) {}


struct Match
{
	Match()
	{
	}

	Match(int x, int y, float similarity, const std::string& class_id, int template_id);

	/// Sort matches with high similarity to the front
	bool operator<(const Match& rhs) const
	{
		// Secondarily sort on template_id for the sake of duplicate removal
		if (similarity != rhs.similarity)
			return similarity > rhs.similarity;
		else
			return template_id < rhs.template_id;
	}

	bool operator==(const Match& rhs) const
	{
		return x == rhs.x && y == rhs.y && similarity == rhs.similarity && class_id == rhs.class_id;
	}

	int x;
	int y;
	float similarity;
	std::string class_id;
	int template_id;
};
inline  Match::Match(int _x, int _y, float _similarity, const std::string& _class_id, int _template_id)
	: x(_x), y(_y), similarity(_similarity), class_id(_class_id), template_id(_template_id)
{
}
