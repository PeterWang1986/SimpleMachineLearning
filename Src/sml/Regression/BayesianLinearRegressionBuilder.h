#ifndef	_BAYESIAN_LINEAR_REGRESSION_BUILDER_H
#define _BAYESIAN_LINEAR_REGRESSION_BUILDER_H

#include "sml/Regression/BayesianLinearRegressionModel.h"

namespace SML {

	class BayesianLrRegressionBuilder
	{
	public:
		BayesianLrRegressionBuilder();
		BayesianLrRegressionBuilder(double a, double b);
		~BayesianLrRegressionBuilder();

		void training(size_t rows,
					size_t cols,
					const double* data,
					const double* target);

		inline void setAlpha(double a) { m_model.setAlpha(a); }
		inline void setBeta(double b) { m_model.setBeta(b); }
		inline void addBasisFunction(BasisFunctionBase* func) { m_model.addBasisFunction(func); }
		void addBiasFunction();
		void addLinearBasisFunction(size_t pos);
		void addSigmodBasisFunction(size_t pos, double mean, double variance);
		void addSingleVarGussian(size_t pos, double mean, double variance);
		void addGussianBasisFunction(size_t dimension, const double* mean, const double* covariance);

		const BayesianLrRegressionModel& getBayesianLinearRegressionModel() const {
			return m_model;
		}

	private:
		void useDefaultBasisFunctions(size_t dimension);

	private:
		BayesianLrRegressionModel	m_model;
	};


}	//end of SML

#endif