#ifndef	_BINARY_LOGISTIC_REGRESSION_H
#define _BINARY_LOGISTIC_REGRESSION_H

#include <vector>

namespace SML {

	class BiLogisticRegression
	{
		typedef	BiLogisticRegression		Self;

		BiLogisticRegression(const Self& rhs);
		Self& operator= (const Self& rhs);

	public:
		BiLogisticRegression();
		~BiLogisticRegression();

		bool stochasticGradientTraining(unsigned int rows, unsigned int cols,
										const double* data, const int* label,
										double learningRate,
										unsigned int maxIteration);
		void squence_training(unsigned int dimension, const double* data,
							int label, double learningRate);

		const std::vector<double>& getWeights() const { return m_Weights; }

	private:
		void initilizeWeights(unsigned int dimension);
		double dissimilarity(const std::vector<double>& v1, const std::vector<double>& v2) const;

	private:
		std::vector<double>		m_Weights;
	};

}	//end of SML

#endif