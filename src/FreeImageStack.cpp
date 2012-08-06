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


#include "FreeImageStack.h"

#include "Exceptions.h"

#include <FreeImage.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <cmath>


FreeImageStack::FreeImageStack(const std::string & rFileName): sFileName_(rFileName)
                                                             , pImageStack_(0)
                                                             , nWidth_(0)
                                                             , nHeight_(0)
                                                             , pBitmap_32f_(0)
                                                             , nMaxXY_(0)
                                                             , nMaxOffset_(0)
{
            // open the bitmap
    pImageStack_ = FreeImage_OpenMultiBitmap(FIF_TIFF, (sFileName_ + ".tif").c_str(), 
                                             FALSE, // create new
                                             TRUE,  // open read-only
                                             FALSE, // keep all slices in memory
                                             TIFF_DEFAULT);
    NPP_ASSERT_NOT_NULL(pImageStack_);
    NPP_ASSERT_NOT_NULL(slices());
    FIBITMAP * pBitmap = FreeImage_LockPage(pImageStack_, 0);
            // store away the size of the first image
            // this information is later used to insure that all slices
            // accessed are of the same size. if they are not an exception
            // is thrown when such a deviating slice is being accessed
    nWidth_  = FreeImage_GetWidth(pBitmap);
    nHeight_ = FreeImage_GetHeight(pBitmap);
    NPP_ASSERT(FreeImage_GetColorType(pBitmap) == FIC_MINISBLACK);
    NPP_ASSERT(FreeImage_GetBPP(pBitmap) == 8);
    FreeImage_UnlockPage(pImageStack_, pBitmap, FALSE);
}

FreeImageStack::FreeImageStack(const std::string & rFileName, 
                               unsigned int nWidth, unsigned int nHeight): sFileName_(rFileName)
                                                                         , pImageStack_(0)
                                                                         , nWidth_(nWidth)
                                                                         , nHeight_(nHeight)
                                                                         , pBitmap_32f_(0)
                                                                         , nMaxXY_(0)
                                                                         , nMaxOffset_(0)
{
            // open the bitmap
    pImageStack_ = FreeImage_OpenMultiBitmap(FIF_TIFF, (sFileName_ + "3D.tif").c_str(), 
                                             TRUE,  // create new
                                             FALSE, // open read-write
                                             FALSE, // keep all slices in memory
                                             TIFF_DEFAULT);
    NPP_ASSERT_NOT_NULL(pImageStack_);
    
    pBitmap_32f_ = FreeImage_AllocateT(FIT_FLOAT, 
                                       nWidth_, nHeight_, 
                                       32 /* bits per pixel */);
    NPP_ASSERT_NOT_NULL(pBitmap_32f_);
    
    nMaxXY_     = std::min(nWidth_, nHeight_);
    nMaxOffset_ = static_cast<unsigned int>(sqrt(2 * static_cast<double>(nMaxXY_-1) * static_cast<double>(nMaxXY_-1)));
    aTimeStepAverages_.reserve(512);
}


FreeImageStack::~FreeImageStack()
{
    FreeImage_Unload(pBitmap_32f_);

    FreeImage_CloseMultiBitmap(pImageStack_, TIFF_DEFAULT);
    
            // if any azimuthal averages have been computed, write
            // them out to disk
    if (aTimeStepAverages_.size() > 0)
    {
        std::ofstream oOutputFile((sFileName_ + "_2DAzimAvg.txt").c_str());
        for (taTimeStepAverages::const_iterator iStep = aTimeStepAverages_.begin();
             iStep != aTimeStepAverages_.end();
             ++iStep)
        {
            for (taAverages::const_iterator iAverage = iStep->begin();
                 iAverage != iStep->end();
                 ++iAverage)
                oOutputFile << *iAverage << " ";
            oOutputFile << "\n";
        }
	    oOutputFile.close();
    }
}

unsigned int
FreeImageStack::slices()
const
{
    return FreeImage_GetPageCount(pImageStack_);
}

unsigned int
FreeImageStack::width()
const
{   
    return nWidth_;
}

unsigned int
FreeImageStack::height()
const
{
    return nHeight_;
}

void
FreeImageStack::loadImage(unsigned int iSlice, npp::ImageNPP_8u_C1 & rImage)
const
{
    NPP_ASSERT_MSG(iSlice < slices(), "Slice index exceeded number of slices in stack.");
    FIBITMAP * pBitmap = FreeImage_LockPage(pImageStack_, iSlice);
    NPP_ASSERT_NOT_NULL(pBitmap);
            // make sure this is an 8-bit single channel image
    NPP_DEBUG_ASSERT(FreeImage_GetColorType(pBitmap) == FIC_MINISBLACK);
    NPP_DEBUG_ASSERT(FreeImage_GetBPP(pBitmap) == 8);
    
    NPP_DEBUG_ASSERT(FreeImage_GetWidth(pBitmap) == nWidth_);
    NPP_DEBUG_ASSERT(FreeImage_GetHeight(pBitmap) == nHeight_);
    unsigned int    nSrcPitch = FreeImage_GetPitch(pBitmap);
    unsigned char * pSrcData  = FreeImage_GetBits(pBitmap);
    
    if (rImage.width() == nWidth_ && rImage.height() == nHeight_)
    {
        NPP_CHECK_CUDA(cudaMemcpy2D(rImage.data(), rImage.pitch(), pSrcData, nSrcPitch, 
                                    nWidth_, nHeight_, cudaMemcpyHostToDevice));
    }
    else
    {
                // create new NPP image
        npp::ImageNPP_8u_C1 oImage(nWidth_, nHeight_);
                // transfer slice data into new device image
        NPP_CHECK_CUDA(cudaMemcpy2D(oImage.data(), oImage.pitch(), pSrcData, nSrcPitch, 
                                    nWidth_, nHeight_, cudaMemcpyHostToDevice));
                // swap the result image with the reference passed into this method
        rImage.swap(oImage);
    }
                // release locked slice
    FreeImage_UnlockPage(pImageStack_, pBitmap, FALSE);
}
    
void
FreeImageStack::appendImage(const npp::ImageNPP_32f_C1 & rImage)
{
    NPP_ASSERT(rImage.width() == nWidth_);
    NPP_ASSERT(rImage.height() == nHeight_);
    
            // create the result image storage using FreeImage so we can easily 
            // save
    unsigned int nResultPitch   = FreeImage_GetPitch(pBitmap_32f_);
    float * pResultData = reinterpret_cast<float *>(FreeImage_GetBits(pBitmap_32f_));

    NPP_CHECK_CUDA(cudaMemcpy2D(pResultData, nResultPitch, rImage.data(), rImage.pitch(),
                                nWidth_ * 4, nHeight_, cudaMemcpyDeviceToHost));
    FreeImage_AppendPage(pImageStack_, pBitmap_32f_);

    appendAzimuthalAnalysis(pResultData, nResultPitch);
}

void 
FreeImageStack::appendAzimuthalAnalysis(const float * pPixels, unsigned int nPitch)
{
            // make the maximum offset the smaller of either width or height of
            // the fourier image
    std::vector<float> aValues(nMaxOffset_ + 1, 0.0f);
    std::vector<unsigned int> aCount(nMaxOffset_ + 1, 0);
    const unsigned char * pData = reinterpret_cast<const unsigned char *>(pPixels);

    for (unsigned int iRow = 0; iRow < nMaxXY_; ++iRow)
        for (unsigned int iColumn = 0; iColumn < nMaxXY_; ++iColumn)
        {
            unsigned int iBin = static_cast<unsigned int>(sqrt(static_cast<double>(iRow) * static_cast<double>(iRow) 
                                                             + static_cast<double>(iColumn) * static_cast<double>(iColumn)));
            aCount[iBin]++;
			 aValues[iBin] += *reinterpret_cast<const float *>((pData + iRow * nPitch + iColumn * sizeof(float)));

        }
    std::vector<float>::iterator iValue;
    std::vector<unsigned int>::const_iterator iCount;
    for (iValue = aValues.begin(), iCount = aCount.begin();
         iValue != aValues.end();
         ++iValue, ++iCount)
        if (*iCount > 0)
            *iValue /= static_cast<float>(*iCount);
        
    aTimeStepAverages_.push_back(aValues);
}


