/**
 * @file  ReadMedicalImage.cxx
 * @brief MEX function to read image using ITK IO classes.
 *
 * Copyright (c) 2011, 2012 University of Pennsylvania. All rights reserved.
 * See http://www.rad.upenn.edu/sbia/software/license.html or COPYING file.
 *
 * Contact: SBIA Group <sbia-software at uphs.upenn.edu>
 */

#include <memory>
using std::auto_ptr;

#include "mex.h" 

#include "ReadMedicalImagePipeline.h"


void mexFunction(int nlhs, mxArray *plhs[],
    int nrhs, const mxArray *prhs[])
{
  // input argument check
  if ((nrhs != 1) || 
      !mxIsChar(prhs[0]) ||
      (nrhs == 2 && !mxIsNumeric(prhs[1])) )
    mexErrMsgTxt("Incorrect arguments, see 'help readmedicalimage'");

  char* filepath = mxArrayToString(prhs[0]);

  try
    {
    auto_ptr<ReadMedicalImagePipeline> pipeline(new ReadMedicalImagePipeline(filepath));

    plhs[0] = mxCreateNumericArray(pipeline->m_numDimensions, pipeline->m_dims,  mxDOUBLE_CLASS, mxREAL);

    double* image = (double *)(mxGetData(plhs[0]));
    double* origin=0, *spacing=0;
    double* direction=0;
    switch(nlhs)
      {
    case 1:    // just image data
      pipeline->CopyAndTranspose(image);
      break;
    case 2:    // image data and origin information
      plhs[1] = mxCreateDoubleMatrix(pipeline->m_numDimensions, 1, mxREAL);
      origin = static_cast<double*>(mxGetPr(plhs[1]));
      pipeline->CopyAndTranspose(image, origin);
      break;
    case 3:    // image data, origin and spacing information
      plhs[1] = mxCreateDoubleMatrix(pipeline->m_numDimensions, 1, mxREAL);
      origin = static_cast<double*>(mxGetPr(plhs[1]));
      plhs[2] = mxCreateDoubleMatrix(pipeline->m_numDimensions, 1, mxREAL);
      spacing = static_cast<double*>(mxGetPr(plhs[2]));
      pipeline->CopyAndTranspose(image, origin, spacing);
      break;
    case 4:    // image data, origin, spacing and direction information
      std::cout << "warning: you asked for direction ! we do not use direction in MATLAB except for saving nii file! Values of the direction are just for passing on to the write function and NOT for your use in MATLAB!"  << std::endl ;
      plhs[1] = mxCreateDoubleMatrix(pipeline->m_numDimensions, 1, mxREAL);
      origin = static_cast<double*>(mxGetPr(plhs[1]));
      plhs[2] = mxCreateDoubleMatrix(pipeline->m_numDimensions, 1, mxREAL);
      spacing = static_cast<double*>(mxGetPr(plhs[2]));
      plhs[3] = mxCreateDoubleMatrix(pipeline->m_numDimensions*pipeline->m_numDimensions, 1, mxREAL);
      direction = static_cast<double*>(mxGetPr(plhs[3]));
      pipeline->CopyAndTranspose(image, origin, spacing, direction);
      break;
    default:
      mexErrMsgTxt("Incorrect output arguments.  See 'help @MATLAB_FUNCTION_NAME@'.");
      }
    }
  catch (std::exception& e)
    {
    mexErrMsgTxt(e.what());
    return;
    }
} 


