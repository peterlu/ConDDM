#ifndef NV_FREE_IMAGE_STACK_H
#define NV_FREE_IMAGE_STACK_H
/*
* Copyright 1993-2007 NVIDIA Corporation.  All rights reserved.
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


#include <npp.h>
#include <string>
#include <vector>

#include "ImagesNPP.h"


//
// Forward declarations
//

struct FIMULTIBITMAP;
struct FIBITMAP;

class FreeImageStack
{
public:
    typedef std::vector<float> taAverages;
    typedef std::vector<taAverages> taTimeStepAverages;
    
    // Create an image stack from a file on disk.
    //      Note: This operation opens a file and this file remains
    //  open for the object's lifetime.
    explicit
    FreeImageStack(const std::string & rFileName);
    
    // Create an empty image stack.
    //      Note: This operation opens a file and this file remains
    //  open for the object's lifetime.
    explicit
    FreeImageStack(const std::string & rFileName, unsigned int nWidth, unsigned int nHeight);
    
    // Destroy the image stack and release all its related resources.
    //      Note: In particular, this destructor closes the stack file.
    virtual
   ~FreeImageStack();
   
    // Number of slices (pages, images) in this stack.
    unsigned int
    slices()
    const;
    
    // Image width
    unsigned int
    width()
    const;
    
    // Image height
    unsigned int
    height()
    const;
    
    void
    loadImage(unsigned int iSlice, npp::ImageNPP_8u_C1 & rImage)
    const;
    
    void
	appendImage(const npp::ImageNPP_32f_C1 & rImage);
	
private:
	void 
	appendAzimuthalAnalysis(const float * pPixels, unsigned int nPitch);
    
    std::string sFileName_;
	FIMULTIBITMAP * pImageStack_;
	unsigned int nWidth_;
	unsigned int nHeight_;
	
	FIBITMAP * pBitmap_32f_;
	
	unsigned int nMaxXY_;
    unsigned int nMaxOffset_;
	taTimeStepAverages aTimeStepAverages_;
};

#endif // NV_FREE_IMAGE_STACK_H