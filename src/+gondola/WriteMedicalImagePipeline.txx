/**
 * @file  WriteMedicalImagePipeline.txx
 * @brief Implements ITK pipeline to write medical image.
 *
 * Copyright (c) 2011, 2012 University of Pennsylvania. All rights reserved.
 * See http://www.rad.upenn.edu/sbia/software/license.html or COPYING file.
 *
 * Contact: SBIA Group <sbia-software at uphs.upenn.edu>
 */

#pragma once
#ifndef _WRITEMEDICALIMAGEPIPELINE_TXX
#define _WRITEMEDICALIMAGEPIPELINE_TXX


#include "WriteMedicalImagePipeline.h"


template <typename PixelType>
WriteMedicalImagePipeline<PixelType>::
WriteMedicalImagePipeline(char* filepath):
  m_filepath(filepath)
{
  m_writer = WriterType::New();
  m_writer->SetFileName(filepath);
  
}

template<typename PixelType>
void 
WriteMedicalImagePipeline<PixelType>
::CopyAndTranspose(const double* image, double* dims, const double *origin, const double *spacing, const double  *direction)
{
  typename ImageType::Pointer itk_image = ImageType::New() ;
  typename ImageType::RegionType region ;
  typename ImageType::SizeType size ;
  size[0] = dims[1] ;    // in MATLAB: the first item is number of rows 
  size[1] = dims[0] ;    // in MATLAB: the second item is number of columns
  size[2] = dims[2] ;
  typename ImageType::IndexType start;
  start[0] = 0 ;
  start[1] = 0 ;
  start[2] = 0 ;
  region.SetSize(size) ;
  region.SetIndex(start) ;
  itk_image->SetRegions( region );
  itk_image->Allocate();

  typename ImageType::SpacingType itk_spacing;
  itk_spacing[0] = spacing[1] ;    // in MATLAB: the first item is number of rows 
  itk_spacing[1] = spacing[0] ;    // in MATLAB: the second item is number of columns
  itk_spacing[2] = spacing[2] ;
  itk_image->SetSpacing( itk_spacing );

  typename ImageType::PointType   itk_origin ;
  itk_origin[0] = origin[1] ;     // in MATLAB: the first item is number of rows 
  itk_origin[1] = origin[0] ;     // in MATLAB: the second item is number of columns
  itk_origin[2] = origin[2] ;
  itk_image->SetOrigin(itk_origin) ;


  typename   ImageType::DirectionType    itk_direction ;
  int cnt = 0 ;
  for (int col=0; col < Dimension; col++)
  {
      for (int row=0; row < Dimension; row++)
      {
          itk_direction(row,col) =  direction[cnt]  ;
          cnt++ ; 
      }
  } 
  itk_image->SetDirection(itk_direction) ;
 

  typedef itk::ImageRegionIterator<ImageType>   IteratorType;
  IteratorType imageIt(itk_image, region);
  unsigned long int count = 0;
  for(imageIt.GoToBegin(); !imageIt.IsAtEnd(); ++imageIt, count++)
  {
     typename ImageType::PixelType  pixelValue ;
     // Kayhan: I hate C to FORTRAN conversion (or otherway around) !
     unsigned long int deptIdx = count/(size[0]*size[1]) ;
     unsigned long int rowIdx  = (count % (size[0]*size[1]))/size[0] ;
     unsigned long int colIdx = (count % (size[0]*size[1])) % size[0] ;

     pixelValue = (PixelType)(image[ rowIdx + colIdx*size[1] + deptIdx*(size[0]*size[1]) ] ) ;
     imageIt.Set(pixelValue);
  }
  
  m_writer->SetInput( itk_image ) ; 

  try
  {
    m_writer->Update();
  }
  catch( itk::ExceptionObject & excp )
  {
    std::cerr << "Error writing the image" << std::endl;
    std::cerr << excp << std::endl;
  }

}


#endif // _WRITEMEDICALIMAGEPIPELINE_H

