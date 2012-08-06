/*
* Copyright 2009-2012 NVIDIA Corporation.  All rights reserved.
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

/* Written by Frank Jargstorff and Peter J. Lu
 * (C)opyright 2009-2012 Frank Jargstorff and Peter J. Lu
 *
 * If you use this code in your research, please cite the original paper:
 * Peter J. Lu (陸述義), Fabio Giavazzi, Thomas E. Angelini, Emanuela Zaccarelli,
 * Frank Jargstorff, Andrew B. Schofield, James N. Wilking, Mark B. Romanowsky,
 * David A. Weitz, and Roberto Cerbino,
 * "Characterizing Concentrated, Multiply Scattering, and Actively Driven
 * Fluorescent Systems with Confocal Differential Dynamic Microscopy"
 * PHYSICAL REVIEW LETTERS, vo. 108, 218103 (2012)
 *
 * http://www.peterlu.org
 */


#include "FourierImageStack.h"

#include "Exceptions.h"
#include <cuda_runtime.h>


FourierImageStack::FourierImageStack(unsigned int nSlices, 
                                     unsigned int nWidth, 
                                     unsigned int nHeight): nSlices_(nSlices)
                                                          , nWidth_ (nWidth)
                                                          , nHeight_(nHeight)
                                                          , pData_(0)
{
    unsigned int nPixels = nSlices_ * nWidth_ * nHeight_;
    if (nPixels > 0)
    {
        NPP_CHECK_CUDA(cudaMalloc(&pData_, nPixels * sizeof(Npp32fc)));
    }
}

FourierImageStack::~FourierImageStack()
{
    cudaFree(pData_);
}

unsigned int
FourierImageStack::slices()
const
{
    return nSlices_;
}

unsigned int
FourierImageStack::width()
const
{
    return nWidth_;
}

unsigned int
FourierImageStack::height()
const
{
    return nHeight_;
}

unsigned int
FourierImageStack::pitch()
const
{
    return nWidth_ * sizeof(Npp32fc);
}

Npp32fc *
FourierImageStack::data(unsigned int iSlice, unsigned int iColumn, unsigned int iRow)
{
    NPP_DEBUG_ASSERT(iSlice < nSlices_);
    return pData_ + (nWidth_ * (nHeight_ * iSlice + iRow) + iColumn);
}

const 
Npp32fc *
FourierImageStack::data(unsigned int iSlice, unsigned int iColumn, unsigned int iRow)
const
{
    NPP_DEBUG_ASSERT(iSlice < nSlices_);
    return pData_ + (nWidth_ * (nHeight_ * iSlice + iRow) + iColumn);
}

