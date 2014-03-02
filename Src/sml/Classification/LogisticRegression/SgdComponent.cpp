
#include "sml/Classification/LogisticRegression/SgdComponent.h"
#include "sml/Utilities/Math/functions.hpp"
#include "sml/Common/SmlMacro.h"

#include <numeric>

namespace SML {

	BiLogisticRegGradient::BiLogisticRegGradient(size_t numWeights,
												const double* weights,
												const double* data,
												const int* label)
												: m_iNumWeights(numWeights)
												, m_weights(weights)
												, m_data(data)
												, m_label(label)
												, m_gradient(new double[numWeights])
	{
	}

	BiLogisticRegGradient::~BiLogisticRegGradient()
	{
		delete [] m_gradient;
	}

	void BiLogisticRegGradient::calcGradient(size_t pos)
	{
		const double* data = m_data + (pos * m_iNumWeights);
		const double a = std::inner_product(m_weights, m_weights + m_iNumWeights, data, 0.0);
		const double diff = sigmoid(a) - m_label[pos];
		for (size_t i=0; i!=m_iNumWeights; ++i)
		{
			m_gradient[i] = data[i] * diff; 
		}
	}

	GaussNewtonBbprop::GaussNewtonBbprop(size_t numWeights,
										const double* weights,
										const double* data,
										const int* label)
										: m_iNumWeights(numWeights)
										, m_weights(weights)
										, m_data(data)
										, m_label(label)
										, m_DiagonalHessian(new double[numWeights])
	{
	}

	GaussNewtonBbprop::~GaussNewtonBbprop()
	{
		delete [] m_DiagonalHessian;
	}

	void GaussNewtonBbprop::calcDiagonalHessian(size_t pos)
	{
		const double* data = m_data + (pos * m_iNumWeights);
		const double a = std::inner_product(m_weights, m_weights + m_iNumWeights, data, 0.0);
		const double y = sigmoid(a);
		const double cc = y * (1 - y);
		for (size_t i=0; i!=m_iNumWeights; ++i)
		{
			const double d = data[i];
			m_DiagonalHessian[i] = d * d * cc;
		}
	}

	BiLRStandErrorFunction::BiLRStandErrorFunction(size_t rows, size_t cols,
												const double* weights,
												const double* data,
												const int* label)
												: m_iRows(rows)
												, m_iCols(cols)
												, m_weights(weights)
												, m_data(data)
												, m_label(label)
	{
	}

	/**
	* maybe can be parallel
	**/
	double BiLRStandErrorFunction::operator() (void) const
	{
		const double penalty = 0 - (std::numeric_limits<double>::max() / (2*m_iRows));
		double sum = 0.0;
		for (size_t i=0; i!=m_iRows; ++i)
		{
			const double* data = m_data + (i * m_iCols);
			const double a = std::inner_product(m_weights, m_weights + m_iCols, data, 0.0);
			const double y = sigmoid(a);
			const double cy = 1 - y;
			const int t = m_label[i];

			if (y < COMMON_DOUBLE_ZERO)
			{
				if (1 == t)
				{
					sum += penalty;
				}
			}
			else if (cy < COMMON_DOUBLE_ZERO)
			{
				if (0 == t)
				{
					sum += penalty;
				}
			}
			else
			{
				sum += (t*log(y) + (1-t)*log(cy));
			}
		}

		return (0 - sum);
	}

}	//end of SML