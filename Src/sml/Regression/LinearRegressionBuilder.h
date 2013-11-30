#ifndef _LINEAR_REGRESSION_BUILDER_H
#define _LINEAR_REGRESSION_BUILDER_H

#include "sml\Regression\LinearRegressionModel.h"

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

		inline void addBasisFunction(BasisFunctionBase* func) { m_model.addBasisFunction(func); }
		inline const LinearRegressionModel& getLinearRegressionModel() const { return m_model; }

	private:
		void useDefaultBasisFunctions(size_t dimension);

	private:
		LinearRegressionModel		m_model;
	};


}	//end of SML

#endif