
# include "ImagePacked.h"

namespace npp{

			template<typename D, size_t N, class A>
			ImagePacked<D, N, A>& ImagePacked<D, N, A>::operator= (const ImagePacked<D, N, A> &rImage)
			{
				// in case of self-assignment
				if (&rImage == this)
				{
					return *this;
				}

				A::Free2D(aPixels_);
				aPixels_ = 0;
				nPitch_ = 0;

				// assign parent class's data fields (width, height)
				Image::operator =(rImage);

				aPixels_ = A::Malloc2D(width(), height(), &nPitch_);
				A::Copy2D(aPixels_, nPitch_, rImage.data(), rImage.pitch(), width(), height());

				return *this;
			}
}