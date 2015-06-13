% Read image list in a COMPARE format
function [V,y,Dim] = ReadDataFromCOMPAREConfigFile(listfile, downsampleratio)
    [imgList, y, Dim, numTissues, dataRoot] = gondola.readImgList(listfile) ;
    y(y==-1) = 2 ;
    [V, Dim] = readImg(imgList, downsampleratio) ;
end

% read image from List
function [V, Dim] = readImg(imgList,DownSampleRatio) 
    % find out the size of the tensor for memory allocation
    fn = imgList{1,1} ; 
    [img,hdr] = gondola.readimage(fn) ;
    % check if it is 4D image (fMRI image) or not
    if (ndims(img)==3)   % 3D image
      Dim = size(img) ; 
      if (DownSampleRatio>1)
          xx1 = linspace(1,size(img,1),size(img,1)) ;
          yy1 = linspace(1,size(img,2),size(img,2)) ;
          zz1 = linspace(1,size(img,3),size(img,3)) ;
          xx2 = linspace(1,size(img,1),size(img,1)/DownSampleRatio) ;
          yy2 = linspace(1,size(img,2),size(img,2)/DownSampleRatio) ;
          zz2 = linspace(1,size(img,3),size(img,3)/DownSampleRatio) ;
          [XX1,YY1,ZZ1] = meshgrid(yy1,xx1,zz1) ;
          [XX2,YY2,ZZ2] = meshgrid(yy2,xx2,zz2) ;
          img2 = interp3(XX1,YY1,ZZ1,double(img),XX2,YY2,ZZ2) ;
          img = img2 ;
          Dim = size(img2) ;
      end
      V = zeros([length(img(:)) size(imgList,1)  size(imgList,2)]) ;
      % read in the data into the tensor
      for jj=1:size(imgList,2)
          for ii=1:size(imgList,1)
              fn = imgList{ii,jj} ;
              [img,hdr] = gondola.readimage(fn) ;
              if (DownSampleRatio>1)
                xx1 = linspace(1,size(img,1),size(img,1)) ;
                yy1 = linspace(1,size(img,2),size(img,2)) ;
                zz1 = linspace(1,size(img,3),size(img,3)) ;
                xx2 = linspace(1,size(img,1),size(img,1)/DownSampleRatio) ;
                yy2 = linspace(1,size(img,2),size(img,2)/DownSampleRatio) ;
                zz2 = linspace(1,size(img,3),size(img,3)/DownSampleRatio) ;
                [XX1,YY1,ZZ1] = meshgrid(yy1,xx1,zz1) ;
                [XX2,YY2,ZZ2] = meshgrid(yy2,xx2,zz2) ;
                img2 = interp3(XX1,YY1,ZZ1,double(img),XX2,YY2,ZZ2) ;
                img = img2 ;
              end
              V(:,ii,jj) = double(img(:)) ;
          end
      end
    elseif (ndims(img)==4)    %fMRI images
      Dim = size(img) ; 
      if (size(imgList,2)~=1) 
         error('If you provide 4D image, there must be only one image per line; i.e. number tissue must be one !! ') ;
      end
      if (DownSampleRatio>=1)
          [XX1,YY1,ZZ1,TT1] = ndgrid( 1:Dim(1), 1:Dim(2), 1:Dim(3), 1:Dim(4));
          [XX2,YY2,ZZ2,TT2] = ndgrid( 1:DownSampleRatio:Dim(1), 1:DownSampleRatio:Dim(2), 1:DownSampleRatio:Dim(3), 1:Dim(4));
          V = zeros([size(XX2,1)*size(XX2,2)*size(XX2,3)    size(imgList,1)  Dim(4)],'single') ;    % we use float format to save some memory
          Dim = [size(XX2,1) size(XX2,2)  size(XX2,3) ]  ;   % update this variable, I need it later on
      else
          error('Upsampling is not supported, read the documentation !!!!') ;
      end
      for ii=1:size(imgList,1)
          fn = imgList{ii,1} ;
          [img,hdr] = gondola.readimage(fn) ;
          img = single(img) ;
          if (DownSampleRatio>1)
             fprintf('Interpolating an image : [%s]  .......',fn) ;
             img2 = interpn(XX1,YY1,ZZ1,TT1,img,XX2,YY2,ZZ2,TT2) ;
             img = single(img2) ;
             fprintf('Done ! \n') ;
          end 
          V(:,ii,:) = reshape(img,[size(V,1) 1 size(V,3)]) ;
      end
    else   % images that 5D and beyond are not supported
       error('images in the list must be either 3D or 4D. Other dimensionalities are not supported !!!') ;
    end
end

% Copyright (c) 2011, 2012 University of Pennsylvania. All rights reserved.
% See http://www.rad.upenn.edu/sbia/software/license.html or COPYING file.
%
% Contact: SBIA Group <sbia-software at uphs.upenn.edu>
