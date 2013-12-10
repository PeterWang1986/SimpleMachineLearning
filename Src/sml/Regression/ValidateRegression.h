#ifndef	_VALIDATE_REGRESSION_H
#define _VALIDATE_REGRESSION_H

#include "sml/Common/SmlMacro.h"

#include <numeric>

namespace SML {

	template <class ForwardAccessIter>
	double smlCalcMean(ForwardAccessIter beg, ForwardAccessIter end)
	{
		typedef ITER_VALUE_TYPE(ForwardAccessIter)	value_type;

		const size_t len = std::distance(beg, end);
		value_type sum = std::accumulate(beg, end, (value_type)0);

		return sum/static_cast<double>(len);
	}

	class RsquaredValidate
	{
	public:
		/**
		*@para	beg_itr1:	the first iterator of target value of training data
		*@para	beg_itr1:	the end iterator of target value of training data
		*@para	beg_itr2:	the first iterator of scoring data
		**/
		template <class RandomAccessIter1, class RandomAccessIter2>
		ITER_VALUE_TYPE(RandomAccessIter1) validate(RandomAccessIter1 beg_itr1,
												RandomAccessIter1 end_itr1,
												RandomAccessIter2 beg_itr2) const;
	};

	template <class RandomAccessIter1, class RandomAccessIter2>
	ITER_VALUE_TYPE(RandomAccessIter1) RsquaredValidate::validate(RandomAccessIter1 beg_itr1,
																RandomAccessIter1 end_itr1,
																RandomAccessIter2 beg_itr2) const
	{
		typedef	ITER_VALUE_TYPE(RandomAccessIter1)	value_type;

		const double yMean = smlCalcMean(beg_itr1, end_itr1);
		double sampleVariance = 0.0, residualSumSquares = 0.0;
		for (; beg_itr1!=end_itr1; ++beg_itr1, ++beg_itr2)
		{
			const double diff1 = *beg_itr1 - yMean;
			const double diff2 = *beg_itr1 - *beg_itr2;
			sampleVariance += diff1 * diff1;
			residualSumSquares += diff2 * diff2;
		}

		if (sampleVariance < COMMON_DOUBLE_ZERO) {
			return 1.0;
		}

		return (1.0 - residualSumSquares/sampleVariance);
	}


}	//end of SML

#endif