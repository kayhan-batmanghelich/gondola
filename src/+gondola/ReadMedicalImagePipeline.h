/**
 * @file  ReadMedicalImagePipeline.h
 * @brief Implements ITK pipeline to read medical image.
 *
 * Copyright (c) 2011, 2012 University of Pennsylvania. All rights reserved.
 * See http://www.rad.upenn.edu/sbia/software/license.html or COPYING file.
 *
 * Contact: SBIA Group <sbia-software at uphs.upenn.edu>
 */

#ifndef _READMEDICALIMAGEPIPELINE_H
#define _READMEDICALIMAGEPIPELINE_H

#include "itkImageFileReader.h"
#include "itkImageRegionConstIterator.h"
#include <iostream>


#include "mex.h" 

template<typename TImageType>
void ReadFile(std::string filename, typename TImageType::Pointer image)
{
  typedef itk::ImageFileReader<TImageType> ReaderType;
  typename ReaderType::Pointer reader = ReaderType::New();
 
  reader->SetFileName(filename);
  reader->Update();
 
  image->Graft(reader->GetOutput());
  
}

class ReadMedicalImagePipeline
{
public:
  ReadMedicalImagePipeline(char* filepath);
  ~ReadMedicalImagePipeline() ;


  /** 
   * @brief Creates and copies the resulting image and its location information to the given
   * double pointers.  Transpose to address C/Fortran column/row memory
   * ordering.
   * 
   * @param image
   * @param origin
   * @param spacing
   */
  void CopyAndTranspose(double* image, double* origin=0, double* spacing=0, double* direction=0);

  // main MATLAB types
  mwSize m_numDimensions ;
  mwSize *m_dims ;

protected:
  // filter types
  //typedef itk::ImageFileReader<ImageType> ReaderType;
  //ReaderType::Pointer m_reader;
  typedef itk::ImageIOBase::IOComponentType ScalarComponentType;
  ScalarComponentType m_componentType ;

  typedef itk::ImageIOBase::IOPixelType   PixelType ;
  PixelType   m_pixelType ;  


  char* m_filepath;
  

};

#endif // _READMEDICALIMAGEPIPELINE_H
