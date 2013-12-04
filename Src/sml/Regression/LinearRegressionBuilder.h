#ifndef _LINEAR_REGRESSION_BUILDER_H
#define _LINEAR_REGRESSION_BUILDER_H

#include "sml\Regression\LinearRegressionModel.h"
#include "sml\Common\SmlMacro.h"

namespace SML {

	class BasisFunctionBase;

	class LinearRegressionBuilder
	{
	public:
		LinearRegressionBuilder();
		~LinearRegressionBuilder();

		void training(size_t rows, size_t cols,
					const double* data,
					const double* target,
					double alpha=0.0);
		template <class Selection>
		void sequence_training(size_t rows,
							size_t cols,
							const double* data,
							const double* target,
							double alpha,
							size_t window,
							size_t MaxIteration);
		void sequence_training(size_t dimension,
							const double* data,
							const double target,
							double alpha);

		inline void addBasisFunction(BasisFunctionBase* func) { m_model.addBasisFunction(func); }
		inline const LinearRegressionModel& getLinearRegressionModel() const { return m_model; }
		void addBiasFunction();
		void addLinearBasisFunction(size_t pos);
		void addSigmodBasisFunction(size_t pos, double mean, double variance);
		void addSingleVarGussian(size_t pos, double mean, double variance);
		void addGussianBasisFunction(size_t dimension, const double* mean, const double* covariance);

	private:
		void useDefaultBasisFunctions(size_t dimension);
		double dissimilarity(const std::vector<double>& v1, const std::vector<double>& v2) const;

	private:
		LinearRegressionModel		m_model;
	};

	template <class Selection>
	void LinearRegressionBuilder::sequence_training(size_t rows,
													size_t cols,
													const double* data,
													const double* target,
													double alpha,
													size_t window,
													size_t MaxIteration)
	{
		if (0 == m_model.getNumBasisFunctions()) {
			useDefaultBasisFunctions(cols);
		}

		const size_t num = m_model.getNumBasisFunctions();
		std::vector<double> weights1(m_model.getWeights());

		size_t count = 0;
		Selection sel(window, 0, rows);
		while (count < MaxIteration)
		{
			while ( sel.isContinue() )
			{
				const size_t pos = sel.getNext();
				sequence_training(cols, data+(pos*cols), *(target+pos), alpha);
			}
			std::vector<double> weights2(m_model.getWeights());
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
	}


}	//end of SML

#endif