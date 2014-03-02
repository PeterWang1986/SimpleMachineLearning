#ifndef _STOP_CRITERIONS_H
#define _STOP_CRITERIONS_H

#include "sml/Common/SmlMacro.h"
//#include <iostream>	//test

namespace Optimization {

	template <class OpCriterion>
	class RelativeDiffValue
	{
	public:
		RelativeDiffValue(double threshold,
						size_t maxIterators,
						const OpCriterion& op);
		RelativeDiffValue(double previous,
						double threshold,
						size_t maxIterators,
						const OpCriterion& op);

		bool isStop();

	private:
		size_t				m_iTimes;
		double				m_fPrevious;
		const double		m_fExitThreshold;
		const size_t		m_iMaxIterators;
		const OpCriterion&	m_functor;
	};

	template <class OpCriterion>
	RelativeDiffValue<OpCriterion>::RelativeDiffValue(double threshold,
													size_t maxIterators,
													const OpCriterion& op)
													: m_iTimes(0)
													, m_fPrevious(-1.0)
													, m_fExitThreshold(threshold)
													, m_iMaxIterators(maxIterators)
													, m_functor(op)
	{
	}

	template <class OpCriterion>
	RelativeDiffValue<OpCriterion>::RelativeDiffValue(double previous,
													double threshold,
													size_t maxIterators,
													const OpCriterion& op)
													: m_iTimes(1)
													, m_fPrevious(previous)
													, m_fExitThreshold(threshold)
													, m_iMaxIterators(maxIterators)
													, m_functor(op)
	{
	}

	template <class OpCriterion>
	bool RelativeDiffValue<OpCriterion>::isStop()
	{
		if ((m_iTimes++) > m_iMaxIterators) return true;

		const double cur = m_functor();
		if (std::fabs(m_fPrevious) > COMMON_DOUBLE_ZERO)
		{
			const double r = std::fabs((cur - m_fPrevious) / m_fPrevious);
			//const double r = std::fabs(cur - m_fPrevious);
			m_fPrevious = cur;
			return (r < m_fExitThreshold);
		}
		else
		{
			m_fPrevious = cur;
			return (std::fabs(cur) < COMMON_DOUBLE_ZERO);
		}

		return true;
	}

}	//end of Optimization

#endif