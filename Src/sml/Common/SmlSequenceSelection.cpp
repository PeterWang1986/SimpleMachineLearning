
#include "sml/Common/SmlSequenceSelection.h"

#include <iostream>
#include <ctime>

namespace SML {

	ForwardSelection::ForwardSelection(size_t window,
									size_t beg,
									size_t end)
									: m_beg(0)
									, m_end(0)
									, m_iWindow((end-beg)>window ? window : (end-beg))
									, m_iBegin(beg)
									, m_iEnd(end)
	{
		m_beg = m_iBegin;
		m_end = m_beg + m_iWindow;
	}

	void ForwardSelection::update()
	{
		m_beg = m_end;
		const size_t offset = m_iEnd - m_end;
		if (offset > m_iWindow)
		{
			m_end += m_iWindow;
		}
		else
		{
			m_end += offset;
		}

		if (m_beg == m_end)
		{
			m_beg = m_iBegin;
			m_end = m_beg + m_iWindow;
		}
	}


	RandomSelection::RandomSelection(size_t window,
									size_t beg,
									size_t end)
									: m_count(0)
									, m_iWindow(window)
									, m_iRange(end - beg)
									, m_iBegin(beg)
	{
		std::srand(static_cast<unsigned int>(std::time(0)));
	}

	size_t RandomSelection::getNext()
	{
		++m_count;
		const size_t n = (std::rand() % m_iRange) + m_iBegin;

		return n;
	}



}	//end of SML