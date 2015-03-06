

#include "Exceptions.h"

namespace npp
{
	extern std::ostream &
    operator << (std::ostream &rOutputStream, const Exception &rException)
    {
        rOutputStream << rException.toString();
        return rOutputStream;
    }
}