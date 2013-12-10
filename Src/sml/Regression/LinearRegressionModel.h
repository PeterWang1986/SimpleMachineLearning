#ifndef	_LINEAR_REGRESSION_MODEL_H
#define _LINEAR_REGRESSION_MODEL_H

#include <iostream>
#include <vector>
#include <cassert>

#include "sml/Common/SmlExtParallel.h"

namespace SML {

	class BasisFunctionBase;

	class LinearRegressionModel
	{
		typedef	LinearRegressionModel		Self;

		LinearRegressionModel(const Self& rhs);
		Self& operator= (const Self& rhs);

	public:
		LinearRegressionModel();
		~LinearRegressionModel();

		void addBasisFunction(BasisFunctionBase* func);
		bool setBasisFunction(size_t pos, BasisFunctionBase* func);
		bool setWeights(size_t pos, double weight);
		void calcDesignMatrix(size_t rows, size_t cols,
							size_t dimesion, const double* data, double* matrix);
		double forecast(const double* data, size_t dimension) const;

		inline size_t getNumBasisFunctions() const { return m_BasisFunctions.size(); }
		inline BasisFunctionBase* getBasisFunction(size_t pos);
		inline const BasisFunctionBase* getBasisFunction(size_t pos) const;
		inline double getWeights(size_t pos) const { return m_Weights[pos]; }
		inline const std::vector<double>& getWeights() const { return m_Weights; }

	private:
		std::vector<double>					m_Weights;
		std::vector<BasisFunctionBase*>		m_BasisFunctions;
	};

	BasisFunctionBase* LinearRegressionModel::getBasisFunction(size_t pos)
	{
		assert( pos < this->getNumBasisFunctions() );
		return m_BasisFunctions[pos];
	}

	const BasisFunctionBase* LinearRegressionModel::getBasisFunction(size_t pos) const
	{
		assert( pos < this->getNumBasisFunctions() );
		return m_BasisFunctions[pos];
	}

	namespace sml_impl {
		class CalcDesignMatrixTaskJob
		{
			typedef CalcDesignMatrixTaskJob		Self;

			Self& operator= (const Self& rhs);

		public:
			CalcDesignMatrixTaskJob(size_t rows, size_t cols,
									SmlPartitioner<size_t>& part,
									const double* pTrainingData,
									double* pMatrixResult,
									std::vector<BasisFunctionBase*>& basis);
			~CalcDesignMatrixTaskJob();
			CalcDesignMatrixTaskJob(const Self& rhs);

			void operator() () const;

		private:
			const size_t						m_iRows;
			const size_t						m_iCols;	//dimension for training data
			SmlPartitioner<size_t>&				m_Partitioner;
			const double*						m_pTrainingData;
			double*								m_pMatrixResult;
			std::vector<BasisFunctionBase*>		m_BasisFunctions;
		};
	}	//end of sml_impl
}	//end of SML
#endif