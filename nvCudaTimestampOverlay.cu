/*
 * Copyright (c) 2016-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <time.h>

#include <cuda.h>

#include "customer_functions.h"
#include "cudaEGL.h"
#include "iva_metadata.h"


unsigned long counter = 0L;

/**
  * Dummy custom pre-process API implematation.
  * It just access mapped surface userspace pointer &
  * memset with specific pattern modifying pixel-data in-place.
  *
  * @param sBaseAddr  : Mapped Surfaces pointers
  * @param smemsize   : surfaces size array
  * @param swidth     : surfaces width array
  * @param sheight    : surfaces height array
  * @param spitch     : surfaces pitch array
  * @param nsurfcount : surfaces count
  */
static void
pre_process (void **sBaseAddr,
                unsigned int *smemsize,
                unsigned int *swidth,
                unsigned int *sheight,
                unsigned int *spitch,
                ColorFormat  *sformat,
                unsigned int nsurfcount,
                void ** usrptr)
{
	return;
}

/**
  * Dummy custom post-process API implematation.
  * It just access mapped surface userspace pointer &
  * memset with specific pattern modifying pixel-data in-place.
  *
  * @param sBaseAddr  : Mapped Surfaces pointers
  * @param smemsize   : surfaces size array
  * @param swidth     : surfaces width array
  * @param sheight    : surfaces height array
  * @param spitch     : surfaces pitch array
  * @param nsurfcount : surfaces count
  */
static void
post_process (void **sBaseAddr,
                unsigned int *smemsize,
                unsigned int *swidth,
                unsigned int *sheight,
                unsigned int *spitch,
                ColorFormat  *sformat,
                unsigned int nsurfcount,
                void ** usrptr)
{

	counter++;

	if ( counter % 30 == 0 ) {

		unsigned long long parsedTimestamp = 0LL;

		if (sformat[0] == COLOR_FORMAT_Y8) {

			for ( int I = 0; I < 64; I++ ) {
				char * pixelPtr = (char *)sBaseAddr[0] + (100 * spitch[0]) + 100 + (4 * I);
				if ( *pixelPtr < 128 ) {
					parsedTimestamp |= (0x1LL << (63 - I));
				}
			}

			std::chrono::time_point<std::chrono::system_clock> now = std::chrono::system_clock::now();
			auto duration = now.time_since_epoch();
			unsigned long long micros = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();

			printf("HUH Thirty frames: %llu %llu %lld\n", micros, parsedTimestamp, (long long)(micros - parsedTimestamp));
		}
	}

	return;
}




__global__ void addTimestampOverlayKernel(int * pYPlanePtr, int * pUvPlanePtr, int pitch, unsigned long long ts) {

	int bitNumber = 63 - (( threadIdx.y & 0x000000FF ) >> 2);
	bool bitval = (( 1LL << bitNumber ) & ts ) == 0;

	if ( threadIdx.x <= 3 ) {

		int row = threadIdx.x + 100;
		int column = threadIdx.y + 100;

  		char * pYpixel = (char*)pYPlanePtr + (row * pitch) + column;
		if ( bitval ) *pYpixel = 255; else *pYpixel = 0;
	
	} 
	

	if ( threadIdx.x == 0 ) {

		int row = 50;
		int column = threadIdx.y + 100;

  		char * pUvpixel = (char*)pUvPlanePtr + (row * pitch) + column;
		*pUvpixel = 128;
	}

	if ( threadIdx.x == 1 ) {

		int row = 51;
		int column = threadIdx.y + 100;

  		char * pUvpixel = (char*)pUvPlanePtr + (row * pitch) + column;
		*pUvpixel = 128;
	}

	return;
}




static int addTimestampOverlay(CUdeviceptr pYPlanePtr, CUdeviceptr pUvPlanePtr, int pitch){

    std::chrono::time_point<std::chrono::system_clock> now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    unsigned long long micros = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();

    //printf("Thirty frames: %llu\n", micros);

    dim3 threadsPerBlock(4,256);
    dim3 blocks(1,1);
    addTimestampOverlayKernel<<<blocks,threadsPerBlock>>>((int*)pYPlanePtr, (int*)pUvPlanePtr, pitch, micros);

    return 0;

}






/**
  * Performs CUDA Operations on egl image.
  *
  * @param image : EGL image
  */
static void
gpu_process (EGLImageKHR image, void ** usrptr)
{
  CUresult status;
  CUeglFrame eglFrame;
  CUgraphicsResource pResource = NULL;

  counter++;

  //if ( counter % 30 != 0 ) {
//	  return;
  //}

  cudaFree(0);
  status = cuGraphicsEGLRegisterImage(&pResource, image, CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE);
  if (status != CUDA_SUCCESS) {
    printf("cuGraphicsEGLRegisterImage failed : %d \n", status);
    return;
  }

  status = cuGraphicsResourceGetMappedEglFrame( &eglFrame, pResource, 0, 0);
  if (status != CUDA_SUCCESS) {
    printf ("cuGraphicsSubResourceGetMappedArray failed\n");
  }

  status = cuCtxSynchronize();
  if (status != CUDA_SUCCESS) {
    printf ("cuCtxSynchronize failed \n");
  }

  if (eglFrame.frameType == CU_EGL_FRAME_TYPE_PITCH) {
    if (eglFrame.eglColorFormat == CU_EGL_COLOR_FORMAT_YUV420_SEMIPLANAR) {
      addTimestampOverlay((CUdeviceptr) eglFrame.frame.pPitch[0], (CUdeviceptr) eglFrame.frame.pPitch[1], eglFrame.pitch);
    } else {
      printf ("Invalid eglcolorformat\n");
    }
  } else {
      printf("Invalid frame type!!\n");
  }

  {
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n",
               cudaGetErrorString(cudaerr));
  }

  status = cuCtxSynchronize();
  if (status != CUDA_SUCCESS) {
    printf ("cuCtxSynchronize failed after memcpy \n");
  }

  status = cuGraphicsUnregisterResource(pResource);
  if (status != CUDA_SUCCESS) {
    printf("cuGraphicsEGLUnRegisterResource failed: %d \n", status);
  }


  if ( counter % 30 == 0 ) {
  	std::chrono::time_point<std::chrono::system_clock> nowAfterOverlay = std::chrono::system_clock::now();
  	auto durationAfterOverlay = nowAfterOverlay.time_since_epoch();
  	unsigned long long microsAfterOverlay = std::chrono::duration_cast<std::chrono::milliseconds>(durationAfterOverlay).count();
  	printf("Micros after overlay: %llu\n", microsAfterOverlay);
  }


}

extern "C" void
init (CustomerFunction * pFuncs)
{
  pFuncs->fPreProcess = pre_process;
  pFuncs->fGPUProcess = gpu_process;
  pFuncs->fPostProcess = post_process;
  printf("libnvcuda_timestamp_overlay.so::init(): The video timestamp processing library has been initialized.\n");
}

extern "C" void
deinit (void)
{
  /* deinitialization */
}
