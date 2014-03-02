#ifndef	_FUNCTIONS_H
#define _FUNCTIONS_H

#include <cmath>
#include <iostream>

namespace SML {

	inline double sigmoid(double x)
	{
		return 1/(1 + std::exp(-x));
	}


}	//end of SML

#endif