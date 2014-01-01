
	//BinaryLogisticRegression.cpp
#include "sml/Classification/LogisticRegression/BinaryLogisticRegression.h"
#include "sml/Utilities/Math/functions.hpp"
#include "sml/Common/SmlSequenceSelection.h"
#include "sml/Common/SmlMacro.h"

#include <numeric>
#include <ctime>

namespace SML {
	using std::vector;

	BiLogisticRegression::BiLogisticRegression()
	{
	}

	BiLogisticRegression::~BiLogisticRegression()
	{
	}

	/**
	* NOTE	we assume the first column of data are all the same with value 1
	**/
	bool BiLogisticRegression::stochasticGradientTraining(unsigned int rows, unsigned int cols,
														const double* data, const int* label,
														double learningRate,
														unsigned int maxIteration)
	{
		if (0==data || 0==label) return false;
		
		initilizeWeights(cols);
		const int window = (rows < 500) ? rows : 500;
		//RandomSelection sel(window, 0, rows);
		ForwardSelection sel(window, 0, rows);
		vector<double> weights1(this->getWeights());
		size_t count = 0;
		while (count < maxIteration)
		{
			while ( sel.isContinue() )
			{
				const size_t pos = sel.getNext();
				squence_training(cols, data+(pos*cols), *(label+pos), learningRate );
			}
			std::vector<double> weights2(this->getWeights());
			const double diff = dissimilarity(weights1, weights2);
			if (diff < COMMON_DOUBLE_ZERO) {
				break;
			}
			else
			{
				weights1 = weights2;
				sel.update();
				++count;
			}
		}

		return true;
	}

	void BiLogisticRegression::squence_training(unsigned int dimension, const double* data,
												int label, double learningRate)
	{
		const double a = std::inner_product(m_Weights.begin(), m_Weights.end(), data, 0.0);
		const double y = sigmoid(a);
		learningRate *= (label - y);

		for (unsigned int i=0; i!=dimension; ++i)
		{
			m_Weights[i] += (learningRate * data[i]);
		}
	}

	void BiLogisticRegression::initilizeWeights(unsigned int dimension)
	{
		std::srand((unsigned int)std::time(0));
		m_Weights.reserve(dimension);
		for (unsigned int i=0; i!=dimension; ++i)
		{
			int a = std::rand() % 1000;
			a -= 500;
			m_Weights.push_back(static_cast<double>(a)/10000);
		}
	}

	double BiLogisticRegression::dissimilarity(const vector<double>& v1,
											const vector<double>& v2) const
	{
		const size_t num = v1.size();
		double sum = 0.0;
		for (size_t i=0; i!=num; ++i)
		{
			const double diff = v1[i] - v2[i];
			sum += diff * diff;
		}

		return std::sqrt(sum);
	}


}	//end of SML