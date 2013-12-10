#ifndef	_SML_EXT_PARALLEL_H
#define _SML_EXT_PARALLEL_H

#include "tbb/atomic.h"

namespace SML {

	template <class ValueType>
	class SmlPartitioner
	{
		typedef	SmlPartitioner		Self;

		SmlPartitioner(const Self& rhs);
		Self& operator= (const Self& rhs);

	public:
		SmlPartitioner(ValueType beg, ValueType end);

		bool getNextItem(ValueType& index);

	private:
		const ValueType			m_Beg;
		const ValueType			m_End;
		tbb::atomic<ValueType>	m_Pos;
	};

	template <class ValueType>
	SmlPartitioner<ValueType>::SmlPartitioner(ValueType beg, ValueType end)
											: m_Beg(beg), m_End(end)
	{
		m_Pos = 0;
	}

	template <class ValueType>
	bool SmlPartitioner<ValueType>::getNextItem(ValueType& index)
	{
		index = m_Pos.fetch_and_increment();
		index += m_Beg;

		return (index < m_End) ? true : false;
	}


}	//end of SML
#endif