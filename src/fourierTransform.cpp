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

#include "ImageIO.h"
#include "FreeImageStack.h"
#include "FourierImageStack.h"

#include "ImagesNPP.h"
#include "Exceptions.h"
#include "StopWatch.h"

#include <cuda_runtime.h>

#include <npp.h>
#include <cufft.h>

#include <FreeImage.h>
#include <tclap/CmdLine.h>

#include <iostream>
#include <sstream>
#include <limits>


class FrugalAllocator_32f_C1
{
public:
    static 
    Npp32f * 
    Malloc2D(unsigned int nWidth, unsigned int nHeight, unsigned int * pPitch)    
    {
        NPP_ASSERT(nWidth * nHeight > 0);
        Npp32f * pResult;
        *pPitch = nWidth *sizeof(float);
        NPP_CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&pResult), (*pPitch) * nHeight));
        NPP_ASSERT(pResult != 0);
        
        return pResult;
    };

    static
    void
    Free2D(Npp32f * pPixels)    
    {
        cudaFree(pPixels);
    };
    
    static
    void
    Copy2D(Npp32f * pDst, size_t nDstPitch, const Npp32f * pSrc, size_t nSrcPitch, size_t nWidth, size_t nHeight)
    {
        cudaError_t eResult;
        eResult = cudaMemcpy2D(pDst, nDstPitch, pSrc, nSrcPitch, nWidth * sizeof(Npp32f), nHeight, cudaMemcpyDeviceToDevice);
        NPP_ASSERT(cudaSuccess == eResult);
    };

    static
    void
    HostToDeviceCopy2D(Npp32f * pDst, size_t nDstPitch, const Npp32f * pSrc, size_t nSrcPitch, size_t nWidth, size_t nHeight)
    {
        cudaError_t eResult;
        eResult = cudaMemcpy2D(pDst, nDstPitch, pSrc, nSrcPitch, nWidth * sizeof(Npp32f), nHeight, cudaMemcpyHostToDevice);
        NPP_ASSERT(cudaSuccess == eResult);
    };

    static
    void
    DeviceToHostCopy2D(Npp32f * pDst, size_t nDstPitch, const Npp32f * pSrc, size_t nSrcPitch, size_t nWidth, size_t nHeight)
    {
        cudaError_t eResult;
        eResult = cudaMemcpy2D(pDst, nDstPitch, pSrc, nSrcPitch, nWidth * sizeof(Npp32f), nHeight, cudaMemcpyDeviceToHost);
        NPP_ASSERT(cudaSuccess == eResult);
    };
};


void
transformStack(const FreeImageStack & rImageStack, FourierImageStack & rFourierStack)
{
    unsigned int nMaxSlices = rImageStack.slices();
    if (nMaxSlices > rFourierStack.slices())
        nMaxSlices = rFourierStack.slices();

    NppiSize oSizeROI = {rImageStack.width(), rImageStack.height()};
            // create plan for the FFT
    cufftHandle oPlanCUFFT;
    NPP_CHECK_CUFFT(cufftPlan2d(&oPlanCUFFT, oSizeROI.width, oSizeROI.height, CUFFT_R2C));
            // allocate 32-bit float intermediate image
            // for this image to work with cuFFT, we must have tightly packed pixels.
    npp::ImageNPP<Npp32f, 1, FrugalAllocator_32f_C1> oSource_32f_C1(oSizeROI.width, oSizeROI.height);
    NPP_DEBUG_ASSERT(oSource_32f_C1.width() * sizeof(Npp32f) == oSource_32f_C1.pitch());
            // allocate 8-bit image 
    npp::ImageNPP_8u_C1 oSource_8u_C1;
    for (unsigned int iSlice = 0; iSlice < nMaxSlices; ++iSlice)
    {
                // load slice
        rImageStack.loadImage(iSlice, oSource_8u_C1);        
                // upconvert 8-bit image to 32-bit float image
        NPP_CHECK_NPP(nppiConvert_8u32f_C1R(oSource_8u_C1.data(),  oSource_8u_C1.pitch(), 
                                            oSource_32f_C1.data(), oSource_32f_C1.pitch(),
                                            oSizeROI));
        NPP_CHECK_CUFFT(cufftExecR2C(oPlanCUFFT, oSource_32f_C1.data(), reinterpret_cast<cufftComplex *>(rFourierStack.data(iSlice))));
    }
}

void
computeDifferenceImage(const FourierImageStack & rFourierImages, 
                       unsigned int nDelta, 
                       npp::ImageNPP_32f_C1 & rResult)
{
    NPP_DEBUG_ASSERT(rResult.width() == rFourierImages.width());
    NPP_DEBUG_ASSERT(rResult.height() == rFourierImages.height());
    
    NppiSize oSizeROI = {rFourierImages.width(), rFourierImages.height()};
    nppiSet_32f_C1R(0.0f, rResult.data(), rResult.pitch(), oSizeROI);

    npp::ImageNPP_32f_C1 oMagnitudeImage(rFourierImages.width(), rFourierImages.height());
    NppiSize oFourierROI = {rFourierImages.width() * 2, rFourierImages.height() * 2};
    npp::ImageNPP_32f_C1 oDifferenceImage(oFourierROI.width, oFourierROI.height);

    for (unsigned int iSlice = 0; iSlice + nDelta < rFourierImages.slices(); ++iSlice)
    {
        NPP_CHECK_NPP(nppiSub_32f_C1R(reinterpret_cast<const Npp32f *>(rFourierImages.data(iSlice + nDelta)), rFourierImages.pitch(),
                                      reinterpret_cast<const Npp32f *>(rFourierImages.data(iSlice)), rFourierImages.pitch(),
                                      oDifferenceImage.data(), oDifferenceImage.pitch(),
                                      oFourierROI));
        NPP_CHECK_NPP(nppiMagnitudeSqr_32fc32f_C1R(reinterpret_cast<const Npp32fc *>(oDifferenceImage.data()), oDifferenceImage.pitch(),
                                                   oMagnitudeImage.data(), oMagnitudeImage.pitch(), 
                                                   oSizeROI));
        NPP_CHECK_NPP(nppiAdd_32f_C1R(oMagnitudeImage.data(), oMagnitudeImage.pitch(),
                                      rResult.data(), rResult.pitch(),
                                      rResult.data(), rResult.pitch(),
                                      oSizeROI));
    }
    
    // scale the result image by a factor
	float scale_factor = 1.0f / (4 * rFourierImages.width() * rFourierImages.width() * (rFourierImages.slices()-nDelta));
    NPP_CHECK_NPP(nppiMulC_32f_C1R(rResult.data(), rResult.pitch(), scale_factor, 
                  rResult.data(), rResult.pitch(), oSizeROI));
}

int 
main(int argc, char *argv[]) 
{
    try
    {
	    TCLAP::CmdLine cmd("Process Stack of Particle Images", ' ',"100125");
        TCLAP::ValueArg<unsigned int> oMaxSlices("s", "slices", "Limit the number of slices processed", 
                                                 false, std::numeric_limits<unsigned int>::max(), "int");
        cmd.add(oMaxSlices);
        TCLAP::ValueArg<unsigned int> oMaxDelta("m", "delta", "Maximum timestep for difference images", 
                                                 false, std::numeric_limits<unsigned int>::max(), "int");
        cmd.add(oMaxDelta);
        TCLAP::ValueArg<unsigned int> oDeltaInc("i", "increment", "Timestep increment", 
                                                 false, 1, "int");
        cmd.add(oDeltaInc);
        
        TCLAP::UnlabeledValueArg<std::string>  oFileStem("filestem", "Multi-page TIFF Image is filestem.tif", true, "", "filestem");
        cmd.add(oFileStem);
        cmd.parse(argc, argv);

	    npp::StopWatch oStopWatch;
	    oStopWatch.start();
	    
	    // This block is here so the destructors of the 
	    // FreeImageStack objects get called which triggers the saving
	    // of the output data.
	    {
            FreeImageStack oStack(oFileStem.getValue());
            unsigned int nSlices   = std::min<unsigned int>(oMaxSlices.getValue(), oStack.slices());
            unsigned int nMaxDelta = std::min<unsigned int>(oMaxDelta.getValue(), nSlices);
            
                    // Allocate a "fourier-image stack" to receive the fourier transformed images
            NppiSize oResultSizeROI = {oStack.width()/2 + 1, oStack.height()/2 + 1};
            
            FourierImageStack oFourierImages(nSlices, oResultSizeROI.width, oResultSizeROI.height);

            transformStack(oStack, oFourierImages);
            
            FreeImageStack oResultStack(oFileStem.getValue() + "_ConDDM", oResultSizeROI.width, oResultSizeROI.height);
            npp::ImageNPP_32f_C1 oDifferenceImage(oResultSizeROI.width, oResultSizeROI.height);
            for (unsigned int iDelta = 1; iDelta < nMaxDelta; iDelta += oDeltaInc.getValue())
            {
                computeDifferenceImage(oFourierImages, iDelta, oDifferenceImage);
                oResultStack.appendImage(oDifferenceImage);
            }
        } // ending block before we stop the stop watch to complete data writing
        
        oStopWatch.stop();
        std::cout << "Elapsed time: " << oStopWatch.elapsed() / 1000 << " s" << std::endl;
    }
    catch (npp::Exception & rException)
    {
        std::cerr << "Program error! The following exception occurred: \n";
        std::cerr << rException;
        std::cerr << "\nAborting." << std::endl;
        
        return -1;
    }
    catch (...)
    {
        std::cerr << "Program error! An unknow type of exception occurred. \n";
        std::cerr << "Aborting." << std::endl;
        
        return -1;
    }
    
    return 0;
}
