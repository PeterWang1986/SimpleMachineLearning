#ifndef	_SML_COMMON_FUNCTIONS_H
#define	_SML_COMMON_FUNCTIONS_H

namespace SML {

	template <class ForwardIter>
	void freeMemory(ForwardIter beg, ForwardIter end)
	{
		for (; beg!=end; ++beg) {
			delete (*beg);
		}
	}

	template <class ForwardIter>
	void freeArrayMemory(ForwardIter beg, ForwardIter end)
	{
		for (; beg!=end; ++beg) {
			delete[] (*beg);
		}
	}

}	//end of SML

#endif