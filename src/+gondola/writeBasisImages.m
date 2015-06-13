% This function is used to save basis vector as NIFTI images
function writeBasisImages(datafile, imagelistfile, outputdir)
    B         = [];
    ConstVars = [];
    load(datafile, 'B', 'ConstVars');
    if size(B) == 0
        error(['Failed to load basis vectors from ' datafile ' !!!']);
    end
    if size(ConstVars) == 0
        error(['Failed to load settings from ' datafile ' !!!']);
    end
    if ~isfield(ConstVars, 'D1') || ~isfield(ConstVars, 'D2') || ~isfield(ConstVars, 'D3')
        error(['Missing D1, D2, or D2 parameter stored in ' datafile ' !!!']);
    end
    if ~isfield(ConstVars, 'DownSampleRatio')
        error(['Missing down-sample ratio parameter stored in ' datafile ' !!!']);
    end
    numChan   = size(B, 3);
    numBasis  = size(B, 2);
    dims      = [ConstVars.D1 ConstVars.D2 ConstVars.D3];
    % read the first image to find out the origin and spacing
    imgList   = gondola.readImgList(imagelistfile);
    imgfn     = char(java.io.File(imgList{1,1}).getAbsolutePath());
    [orgImg, origin, PixelDimensions, direction] = gondola.readmedicalimage(imgfn) ;
    orgDirect = reshape(direction,[sqrt(length(direction)) sqrt(length(direction))])' ;
    orgDirect = orgDirect(1:3,1:3) ;    % if the image 4D, we only case about 3D domain
    orgDirect = orgDirect' ;   % converting MATLAB indexing to VNL indexing
    orgDims   = [ size(orgImg,1) size(orgImg,2)  size(orgImg,3) ] ;
    origin    = origin(1:3) ; 
    spacing   = [PixelDimensions(1)  PixelDimensions(2)   PixelDimensions(3) ] ; 
    for chanCnt=1:numChan
        for rCnt=1:numBasis
            fn = sprintf('%s/basis-chan%d-col%d.nii.gz', outputdir, chanCnt, rCnt) ;
            try
                img = reshape(full(double(B(:,rCnt,chanCnt))), dims) ;
                img = fixImgDimension(img, orgDims, ConstVars.DownSampleRatio) ;
                gondola.writemedicalimage(fn, img, origin, spacing, orgDirect(:), 'float') ;
                fprintf('Wrote image: %s\n', fn) ;
            catch exception
                error(['Failed to write image ' fn ' !!! Error: ' exception.message]) ;
            end
        end
    end
end

% this function takes care of down-sampling  up-sampling of the original and down-sampled image
function img = fixImgDimension(img, orgDims, DownSampleRatio)
    if  (DownSampleRatio > 1)
        xx1 = linspace(1,orgDims(1),orgDims(1)) ;
        yy1 = linspace(1,orgDims(2),orgDims(2)) ;
        zz1 = linspace(1,orgDims(3),orgDims(3)) ;
        xx2 = linspace(1,orgDims(1),orgDims(1)/DownSampleRatio) ;
        yy2 = linspace(1,orgDims(2),orgDims(2)/DownSampleRatio) ;
        zz2 = linspace(1,orgDims(3),orgDims(3)/DownSampleRatio) ;
        [XX1,YY1,ZZ1] = meshgrid(yy1,xx1,zz1) ;
        [XX2,YY2,ZZ2] = meshgrid(yy2,xx2,zz2) ;
        img2 = interp3(XX2,YY2,ZZ2,double(img),XX1,YY1,ZZ1) ;
        img = img2 ; 
    elseif (DownSampleRatio == 1)
        if  (size(img)~=orgDims)    % how is that possible !!!
            error('Downsample ratio is one but original image does not match with basis vector !!!') ;
        end
    else
        error(['Invalid downsample ratio: ' num2str(DownSampleRatio)]) ;
    end
end

% Copyright (c) 2011, 2012 University of Pennsylvania. All rights reserved.
% See http://www.rad.upenn.edu/sbia/software/license.html or COPYING file.
%
% Contact: SBIA Group <sbia-software at uphs.upenn.edu>
