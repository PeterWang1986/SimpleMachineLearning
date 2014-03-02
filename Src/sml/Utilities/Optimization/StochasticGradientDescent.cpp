
//StochasticGradientDescent.cpp

#include "sml/Utilities/Optimization/StochasticGradientDescent.hpp"

namespace Optimization {

	DifferenceVector::DifferenceVector(double threshold,
									size_t numWeights,
									const double* weights)
									: m_fExitThreshold(threshold)
									, m_iNumWeights(numWeights)
									, m_previous(new double[numWeights])
									, m_currents(weights)
									, m_times(1)
									, m_count(0)
									, m_max_times(std::numeric_limits<size_t>::max())
									, m_max_count(1)
	{
		for (size_t i=0; i!=numWeights; ++i)
		{
			m_previous[i] = m_currents[i] + 1.0;
		}
	}

	DifferenceVector::DifferenceVector(double threshold,
									size_t numWeights,
									const double* weights,
									size_t maxTimes,
									size_t count)
									: m_fExitThreshold(threshold)
									, m_iNumWeights(numWeights)
									, m_previous(new double[numWeights])
									, m_currents(weights)
									, m_times(1)
									, m_count(0)
									, m_max_times(maxTimes)
									, m_max_count(count)
	{
		for (size_t i=0; i!=numWeights; ++i)
		{
			m_previous[i] = m_currents[i] + 1.0;
		}
	}

	DifferenceVector::~DifferenceVector()
	{
		delete [] m_previous;
	}
		
	bool DifferenceVector::isStop()
	{
		if ((m_times++) > m_max_times) return true;

		double sum = 0.0;
		for (size_t i=0; i!=m_iNumWeights; ++i)
		{
			const double diff = m_currents[i] - m_previous[i];
			sum += diff * diff;
			m_previous[i] = m_currents[i];
		}

		if (std::sqrt(sum) < m_fExitThreshold)
		{
			if (m_count < m_max_count)
			{
				++m_count;
			}
			else
			{
				return true;
			}
		}
		else
		{
			m_count = 0;
		}

		return false;	
	}


}	//end of Optimization