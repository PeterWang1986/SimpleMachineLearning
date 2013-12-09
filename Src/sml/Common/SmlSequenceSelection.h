#ifndef	_SML_SQUENCE_SELECTION_H
#define _SML_SQUENCE_SELECTION_H

namespace SML {

	/**
	*NOTE:	Not threads safe
	**/
	class ForwardSelection
	{
		typedef	ForwardSelection		Self;

	public:
		ForwardSelection(size_t window, size_t beg, size_t end);

		inline bool isContinue() const { return (m_beg != m_end); }
		inline size_t getNext() { return m_beg++; }
		void update();

	private:
		size_t			m_beg;
		size_t			m_end;
		const size_t	m_iWindow;
		const size_t	m_iBegin;
		const size_t	m_iEnd;
	};

	class RandomSelection
	{
		typedef	RandomSelection			Self;

	public:
		RandomSelection(size_t window, size_t beg, size_t end);

		inline bool isContinue() const { return (m_count < m_iWindow); }
		size_t getNext();
		inline void update() { m_count = 0; }

	private:
		size_t			m_count;
		const size_t	m_iWindow;
		const size_t	m_iRange;
		const size_t	m_iBegin;
	};


}	//end of SML

#endif