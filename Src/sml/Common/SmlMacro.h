#ifndef	_SML_MACRO_H
#define _SML_MACRO_H

#include <iterator>

namespace SML {

#define	COMMON_DOUBLE_ZERO		0.0000001

#define ITER_VALUE_TYPE(itr)	typename std::iterator_traits<itr>::value_type


}	//end of SML
#endif