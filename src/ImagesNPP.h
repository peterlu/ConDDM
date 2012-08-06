#ifndef NV_UTIL_NPP_IMAGES_NPP_H
#define NV_UTIL_NPP_IMAGES_NPP_H
/*
* Copyright 2008-2009 NVIDIA Corporation.  All rights reserved.
*
* NOTICE TO USER:
*
* This source code is subject to NVIDIA ownership rights under U.S. and
* international Copyright laws.
*
* NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
* CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
* IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
* REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
* MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
* IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
* OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
* OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
* OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE
* OR PERFORMANCE OF THIS SOURCE CODE.
*
* U.S. Government End Users.  This source code is a "commercial item" as
* that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of
* "commercial computer software" and "commercial computer software
* documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995)
* and is provided to the U.S. Government only as a commercial end item.
* Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
* 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
* source code with only those rights set forth herein.
*/

#include "Exceptions.h"
#include "ImagePacked.h"

#include "ImageAllocatorsNPP.h"
#include <cuda_runtime.h>

namespace npp
{
            // forward declaration
    template<typename D, size_t N, class A> class ImageCPU;

    template<typename D, size_t N, class A>
    class ImageNPP: public npp::ImagePacked<D, N, A>
    {
    public:
        ImageNPP()
        { ; }
        
        ImageNPP(unsigned int nWidth, unsigned int nHeight): ImagePacked<D, N, A>(nWidth, nHeight)
        { ; }

        ImageNPP(const npp::Image::Size & rSize): ImagePacked<D, N, A>(rSize)
        { ; }

        ImageNPP(const ImageNPP<D, N, A> & rImage): Image(rImage)
        { ; }
        
        template<class X>
        explicit
        ImageNPP(const ImageCPU<D, N, X> & rImage): ImagePacked<D, N, A>(rImage.width(), rImage.height())
        {
            A::HostToDeviceCopy2D(ImagePacked<D, N, A>::data(), ImagePacked<D, N, A>::pitch(), rImage.data(), rImage.pitch(), ImagePacked<D, N, A>::width(), ImagePacked<D, N, A>::height());
        }

        virtual
       ~ImageNPP()
        { ; }
        
        ImageNPP &
        operator= (const ImageNPP<D, N, A> & rImage)
        {
            ImagePacked<D, N, A>::operator= (rImage);
            
            return *this;
        }
        
        void
        copyTo(D * pData, unsigned int nPitch)
        const
        {
            NPP_ASSERT((ImagePacked<D, N, A>::width() * sizeof(npp::Pixel<D, N>) <= nPitch));
            A::DeviceToHostCopy2D(pData, nPitch, ImagePacked<D, N, A>::data(), ImagePacked<D, N, A>::pitch(), ImagePacked<D, N, A>::width(), ImagePacked<D, N, A>::height());
        }

	    void
	    copyFrom(D * pData, unsigned int nPitch) 
	    {
		    NPP_ASSERT((ImagePacked<D, N, A>::width() * sizeof(npp::Pixel<D, N>) <= nPitch));
		    A::HostToDeviceCopy2D(ImagePacked<D, N, A>::data(), ImagePacked<D, N, A>::pitch(), pData, nPitch, ImagePacked<D, N, A>::width(), ImagePacked<D, N, A>::height());
	    }
    };

    typedef ImageNPP<Npp8u,  1,   npp::Allocator<Npp8u, 1> >   ImageNPP_8u_C1;
    typedef ImageNPP<Npp8u,  2,   npp::Allocator<Npp8u, 2> >   ImageNPP_8u_C2;
    typedef ImageNPP<Npp8u,  3,   npp::Allocator<Npp8u, 3> >   ImageNPP_8u_C3;
    typedef ImageNPP<Npp8u,  4,   npp::Allocator<Npp8u, 4> >   ImageNPP_8u_C4;

    typedef ImageNPP<Npp16u, 1,   npp::Allocator<Npp16u, 1> >  ImageNPP_16u_C1;
    typedef ImageNPP<Npp16u, 3,   npp::Allocator<Npp16u, 3> >  ImageNPP_16u_C3;
    typedef ImageNPP<Npp16u, 4,   npp::Allocator<Npp16u, 4> >  ImageNPP_16u_C4;

    typedef ImageNPP<Npp16s, 1,   npp::Allocator<Npp16s, 1> >  ImageNPP_16s_C1;
    typedef ImageNPP<Npp16s, 3,   npp::Allocator<Npp16s, 3> >  ImageNPP_16s_C3;
    typedef ImageNPP<Npp16s, 4,   npp::Allocator<Npp16s, 4> >  ImageNPP_16s_C4;

    typedef ImageNPP<Npp32s, 1,   npp::Allocator<Npp32s, 1> >  ImageNPP_32s_C1;
    typedef ImageNPP<Npp32s, 3,   npp::Allocator<Npp32s, 3> >  ImageNPP_32s_C3;
    typedef ImageNPP<Npp32s, 4,   npp::Allocator<Npp32s, 4> >  ImageNPP_32s_C4;

    typedef ImageNPP<Npp32f, 1,   npp::Allocator<Npp32f, 1> >  ImageNPP_32f_C1;
    typedef ImageNPP<Npp32f, 2,   npp::Allocator<Npp32f, 2> >  ImageNPP_32f_C2;
    typedef ImageNPP<Npp32f, 3,   npp::Allocator<Npp32f, 3> >  ImageNPP_32f_C3;
    typedef ImageNPP<Npp32f, 4,   npp::Allocator<Npp32f, 4> >  ImageNPP_32f_C4;

} // npp namespace

#endif // NV_UTIL_NPP_IMAGES_NPP_H
