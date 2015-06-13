/**
 * @file  ReadMedicalImagePipeline.cxx
 * @brief Implements ITK pipeline to read medical image.
 *
 * Copyright (c) 2011, 2012 University of Pennsylvania. All rights reserved.
 * See http://www.rad.upenn.edu/sbia/software/license.html or COPYING file.
 *
 * Contact: SBIA Group <sbia-software at uphs.upenn.edu>
 */

#include "ReadMedicalImagePipeline.h"

#include <sstream>
#include <exception>


ReadMedicalImagePipeline::ReadMedicalImagePipeline(char* filepath)
:
  m_dims(NULL),
  m_filepath(filepath)
{
  itk::ImageIOBase::Pointer imageIO = itk::ImageIOFactory::CreateImageIO(filepath, itk::ImageIOFactory::ReadMode);
 
  if (imageIO.IsNull()) {
    std::ostringstream errmsg;
    errmsg << "Failed to create IO object for image file " << filepath;
    throw std::invalid_argument(errmsg.str());
  }

  imageIO->SetFileName(filepath);
  imageIO->ReadImageInformation();

  m_componentType = imageIO->GetComponentType();
  m_pixelType     = imageIO->GetPixelType();
  m_numDimensions = static_cast<mwSize>(imageIO->GetNumberOfDimensions());
  m_dims          = new mwSize[m_numDimensions];
 
  switch(m_numDimensions) {
     case 2:
        m_dims[0] = static_cast<mwSize>(imageIO->GetDimensions(1) ) ;
        m_dims[1] = static_cast<mwSize>(imageIO->GetDimensions(0) ) ;
        break ;
     case 3:
        m_dims[0] = static_cast<mwSize>(imageIO->GetDimensions(1) ) ;
        m_dims[1] = static_cast<mwSize>(imageIO->GetDimensions(0) ) ;
        m_dims[2] = static_cast<mwSize>(imageIO->GetDimensions(2) ) ;
        break ;
     case 4:
        m_dims[0] = static_cast<mwSize>(imageIO->GetDimensions(1) ) ;
        m_dims[1] = static_cast<mwSize>(imageIO->GetDimensions(0) ) ;
        m_dims[2] = static_cast<mwSize>(imageIO->GetDimensions(2) ) ;
        m_dims[3] = static_cast<mwSize>(imageIO->GetDimensions(3) ) ;
        break ;
     default:
       std::ostringstream errmsg;
       errmsg << "Cannot read image " << filepath << "! Unsupported number of dimensions: " << m_numDimensions;
       throw std::runtime_error(errmsg.str());
  }
}

ReadMedicalImagePipeline::~ReadMedicalImagePipeline() 
{
    if (m_dims) delete m_dims;
}

void ReadMedicalImagePipeline::CopyAndTranspose(double* image, double* origin, double* spacing, double* direction)
{
  switch (m_pixelType) {
    case itk::ImageIOBase::SCALAR:
      if (m_numDimensions==2) {
         typedef itk::Image<double, 2> ImageType;
         ImageType::Pointer itk_image = ImageType::New();
         ReadFile<ImageType>(std::string(m_filepath), itk_image);
         ImageType::RegionType region = itk_image->GetLargestPossibleRegion();
         ImageType::SizeType size = region.GetSize();
         typedef itk::ImageRegionConstIterator<ImageType> ConstIteratorType;
         ConstIteratorType imageIt(itk_image, region);
         unsigned long int count = 0;
         for (imageIt.GoToBegin(); 
              !imageIt.IsAtEnd(); 
              ++imageIt, count++) {
              // Kayhan: I hate C to FORTRAN conversion (or otherway around) !
              unsigned long int rowIdx  = (count )/size[0] ;
              unsigned long int colIdx = (count ) % size[0] ;
    
              image[ rowIdx + colIdx*size[1]  ] = imageIt.Value() ;
         }
         if (origin != 0)
         {
            ImageType::PointType itk_origin = itk_image->GetOrigin();
            origin[0] = static_cast<double>(itk_origin[1]);
            origin[1] = static_cast<double>(itk_origin[0]);
         }
         if (spacing != 0)
         {
             ImageType::SpacingType itk_spacing = itk_image->GetSpacing();
             spacing[0] = static_cast<double>(itk_spacing[1]);
             spacing[1] = static_cast<double>(itk_spacing[0]);
         }
         if (direction != 0)
         {
             ImageType::DirectionType  itk_direction = itk_image->GetDirection() ;
             int cnt = 0 ;
             for (int col=0; col < m_numDimensions; col++)
             {
                 for (int row=0; row < m_numDimensions; row++)
                 {
                     direction[cnt] = itk_direction(row,col); 
                     cnt++ ; 
                 }
             } 
         } 


      }
      else if (m_numDimensions==3)
      {
         typedef itk::Image<double, 3> ImageType;
         ImageType::Pointer itk_image = ImageType::New();
         ReadFile<ImageType>(std::string(m_filepath), itk_image);
         ImageType::RegionType region = itk_image->GetLargestPossibleRegion();
         ImageType::SizeType size = region.GetSize();
         typedef itk::ImageRegionConstIterator<ImageType> ConstIteratorType;
         ConstIteratorType imageIt(itk_image, region);
         unsigned long int count = 0;
         for (imageIt.GoToBegin(); 
              !imageIt.IsAtEnd(); 
              ++imageIt, count++) {
              // Kayhan: I hate C to FORTRAN conversion (or otherway around) !
              unsigned long int deptIdx = count/(size[0]*size[1]) ;
              unsigned long int rowIdx  = (count % (size[0]*size[1]))/size[0] ;
              unsigned long int colIdx = (count % (size[0]*size[1])) % size[0] ;

              image[ rowIdx + colIdx*size[1] + deptIdx*(size[0]*size[1]) ] = imageIt.Value() ;

         }
         if (origin != 0)
         {
            ImageType::PointType itk_origin = itk_image->GetOrigin();
            origin[0] = static_cast<double>(itk_origin[1]);
            origin[1] = static_cast<double>(itk_origin[0]);
            origin[2] = static_cast<double>(itk_origin[2]);
         }
         if (spacing != 0)
         {
             ImageType::SpacingType itk_spacing = itk_image->GetSpacing();
             spacing[0] = static_cast<double>(itk_spacing[1]);
             spacing[1] = static_cast<double>(itk_spacing[0]);
             spacing[2] = static_cast<double>(itk_spacing[2]);
         } 
         if (direction != 0)
         {
             ImageType::DirectionType  itk_direction = itk_image->GetDirection() ;
             int cnt = 0 ;
             for (int col=0; col < m_numDimensions; col++)
             {
                 for (int row=0; row < m_numDimensions; row++)
                 {
                     direction[cnt] = itk_direction(row,col); 
                     cnt++ ; 
                 }
             } 
         } 

      }
      else if (m_numDimensions==4)
      {
         typedef itk::Image<double, 4> ImageType;
         ImageType::Pointer itk_image = ImageType::New();
         ReadFile<ImageType>(std::string(m_filepath), itk_image);
         ImageType::RegionType region = itk_image->GetLargestPossibleRegion();
         ImageType::SizeType size = region.GetSize();
         typedef itk::ImageRegionConstIterator<ImageType> ConstIteratorType;
         ConstIteratorType imageIt(itk_image, region);
         unsigned long int count = 0;
         for (imageIt.GoToBegin(); 
              !imageIt.IsAtEnd(); 
              ++imageIt, count++) {
              // Kayhan: I hate C to FORTRAN conversion (or otherway around) !
              unsigned long int timeIdx = count/(size[0]*size[1]*size[2])  ;
              unsigned long int deptIdx = (count % (size[0]*size[1]*size[2]) ) / (size[0]*size[1]) ;
              unsigned long int rowIdx  = ((count % (size[0]*size[1]*size[2]) ) % (size[0]*size[1])) / size[0] ;
              unsigned long int colIdx =  ((count % (size[0]*size[1]*size[2]) ) % (size[0]*size[1])) % size[0] ;
    
              image[ rowIdx + colIdx*size[1] + deptIdx*(size[0]*size[1]) + timeIdx*(size[0]*size[1]*size[2]) ] = imageIt.Value() ;
         }
         if (origin != 0)
         {
            ImageType::PointType itk_origin = itk_image->GetOrigin();
            origin[0] = static_cast<double>(itk_origin[1]);
            origin[1] = static_cast<double>(itk_origin[0]);
            origin[2] = static_cast<double>(itk_origin[2]);
            origin[3] = static_cast<double>(itk_origin[3]);

         }
         if (spacing != 0)
         {
             ImageType::SpacingType itk_spacing = itk_image->GetSpacing();
             spacing[0] = static_cast<double>(itk_spacing[1]);
             spacing[1] = static_cast<double>(itk_spacing[0]);
             spacing[2] = static_cast<double>(itk_spacing[2]);
             spacing[3] = static_cast<double>(itk_spacing[3]);
         } 
         if (direction != 0)
         {
             ImageType::DirectionType  itk_direction = itk_image->GetDirection() ;
             int cnt = 0 ;
             for (int col=0; col < m_numDimensions; col++)
             {
                 for (int row=0; row < m_numDimensions; row++)
                 {
                     direction[cnt] = itk_direction(row,col); 
                     cnt++ ; 
                 }
             } 
         } 

      }
      else
      {
        std::ostringstream errmsg;
        errmsg << "Unsupported number of image dimensions: " << m_numDimensions;
        throw std::runtime_error(errmsg.str());
      }
      break;
 
    default:
        std::ostringstream errmsg;
        errmsg << "Unsupported data type: " << m_pixelType;
        throw std::runtime_error(errmsg.str());
  }
}
