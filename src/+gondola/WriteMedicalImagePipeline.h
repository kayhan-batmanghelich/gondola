/**
 * @file  WriteMedicalImagePipeline.cxx
 * @brief Implements ITK pipeline to write medical image.
 *
 * Copyright (c) 2011, 2012 University of Pennsylvania. All rights reserved.
 * See http://www.rad.upenn.edu/sbia/software/license.html or COPYING file.
 *
 * Contact: SBIA Group <sbia-software at uphs.upenn.edu>
 */

#ifndef _WRITEMEDICALIMAGEPIPELINE_H
#define _WRITEMEDICALIMAGEPIPELINE_H

#include "itkImageFileWriter.h"
#include "itkImage.h"

#include <iostream>


template <typename PixelType>
class WriteMedicalImagePipeline
{
public:
  WriteMedicalImagePipeline(char* filepath);


  /** 
   * @brief Creates and copies the resulting image and its location information to the given
   * double pointers.  Transpose to address C/Fortran column/row memory
   * ordering.
   * 
   * @param image
   * @param origin
   * @param spacing
   * @param direction
   */
  void CopyAndTranspose(const double* image, double* dims, const double *origin, const double *spacing, const double  *direction);

  // main itk types
  //typedef double PixelType;
  const static unsigned int Dimension = 3;
  typedef typename itk::Image<PixelType, Dimension> ImageType;

protected:
  // filter types
  typedef typename itk::ImageFileWriter<ImageType> WriterType;
  typename WriterType::Pointer m_writer;

  char* m_filepath;
  double m_spacing[3] ;
  double m_origin[3] ;
  double m_direction[9] ;

};


// include template definitions
#include "WriteMedicalImagePipeline.txx"


#endif // _WRITEMEDICALIMAGEPIPELINE_H
