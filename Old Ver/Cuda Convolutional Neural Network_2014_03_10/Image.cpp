
#include "Image.h"

namespace npp
{
	extern bool
    operator== (const Image::Size &rFirst, const Image::Size &rSecond)
    {
        return rFirst.nWidth == rSecond.nWidth && rFirst.nHeight == rSecond.nHeight;
    }

    extern bool
    operator!= (const Image::Size &rFirst, const Image::Size &rSecond)
    {
        return rFirst.nWidth != rSecond.nWidth || rFirst.nHeight != rSecond.nHeight;
    }
}