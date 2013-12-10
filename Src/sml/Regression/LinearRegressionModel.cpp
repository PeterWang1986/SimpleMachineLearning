
#include "sml/Common/SmlCommonFunctions.h"
#include "sml/Regression/BasisFunctions.h"
#include "sml/Regression/LinearRegressionModel.h"

#include "tbb/task_group.h"
#include "tbb/task_scheduler_init.h"

#include <numeric>

namespace SML {
	using std::vector;

	LinearRegressionModel::LinearRegressionModel()
	{
	}

	LinearRegressionModel::~LinearRegressionModel()
	{
		freeMemory(m_BasisFunctions.begin(), m_BasisFunctions.end());
	}

	void LinearRegressionModel::addBasisFunction(BasisFunctionBase* func)
	{
		m_BasisFunctions.push_back(func);
		m_Weights.push_back(0.0);
	}

	bool LinearRegressionModel::setBasisFunction(size_t pos, BasisFunctionBase* func)
	{
		if (pos >= this->getNumBasisFunctions()) return false;

		BasisFunctionBase* pFunc = this->getBasisFunction(pos);
		m_BasisFunctions[pos] = func;
		delete pFunc;

		return true;
	}

	bool LinearRegressionModel::setWeights(size_t pos, double weight)
	{
		if (pos >= this->getNumBasisFunctions()) return false;

		m_Weights[pos] = weight;
		return true;
	}

	/**
	@para rows:			the row number of design matrix
	@para cols:			the column number of design matrix
	@para dimension:	the dimension of vector x (input record)
	@para data:			training data, rows * dimension matrix
	@para matrix:		column-major based matrix (design matrix)
	**/
	void LinearRegressionModel::calcDesignMatrix(size_t rows,
												size_t cols,
												size_t dimension,
												const double* data,
												double* matrix)
	{
		int t = tbb::task_scheduler_init::default_num_threads() >> 1;
		t = (t > 1) ? t : 1;
		const int threads = ((size_t)t > cols) ? cols : t;
		SmlPartitioner<size_t> partitioner(0, cols);
		sml_impl::CalcDesignMatrixTaskJob taskForDesignMatrx(rows, dimension, partitioner,
															data, matrix, m_BasisFunctions);
		tbb::task_group taskGroup;
		for (int i=0; i!=threads; ++i)
		{
			taskGroup.run(taskForDesignMatrx);
		}
		taskGroup.wait();
	}

	double LinearRegressionModel::forecast(const double* data, size_t dimension) const
	{
		const size_t NumWeights = this->getNumBasisFunctions();
		vector<double> phiVectors(NumWeights, 0);
		for (size_t i=0; i!=NumWeights; ++i)
		{
			phiVectors[i] = (this->getBasisFunction(i))->phi(data, dimension);
		}

		return std::inner_product(phiVectors.begin(), phiVectors.end(), m_Weights.begin(), 0.0);
	}

	namespace sml_impl {

		CalcDesignMatrixTaskJob::CalcDesignMatrixTaskJob(size_t rows, size_t cols,
														SmlPartitioner<size_t>& part,
														const double* pTrainingData,
														double* pMatrixResult,
														vector<BasisFunctionBase*>& basis)
														: m_iRows(rows)
														, m_iCols(cols)
														, m_Partitioner(part)
														, m_pTrainingData(pTrainingData)
														, m_pMatrixResult(pMatrixResult)
														, m_BasisFunctions(basis)
		{
		}

		CalcDesignMatrixTaskJob::~CalcDesignMatrixTaskJob()
		{
		}

		CalcDesignMatrixTaskJob::CalcDesignMatrixTaskJob(const Self& rhs)
														: m_iRows(rhs.m_iRows)
														, m_iCols(rhs.m_iCols)
														, m_Partitioner(rhs.m_Partitioner)
														, m_pTrainingData(rhs.m_pTrainingData)
														, m_pMatrixResult(rhs.m_pMatrixResult)
														, m_BasisFunctions(rhs.m_BasisFunctions)
		{
		}

		void CalcDesignMatrixTaskJob::operator() () const
		{
			size_t pos = 0;
			while (m_Partitioner.getNextItem(pos))
			{
				const BasisFunctionBase* basis = m_BasisFunctions[pos];
				double* pColResult = m_pMatrixResult + (m_iRows * pos);
				for (size_t i=0; i!=m_iRows; ++i)
				{
					const double* data = m_pTrainingData + (i * m_iCols);
					pColResult[i] = basis->phi(data, m_iCols);
				}
			}
		}

	}	//end of sml_impl


}	//end of SML