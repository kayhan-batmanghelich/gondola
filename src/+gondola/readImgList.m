% this function is just to read COMPARE image list and provide dimension, number of tissue and datarooot
function [imgList, labels, Dim, numTissues, dataRoot] = readImgList(ListFN) 
    fid = fopen(ListFN,'r') ;
    tline = fgetl(fid) ;
    numImg = sscanf(tline, '%d %d') ;
    numTissues = numImg(2) ;
    numImg = numImg(1) ;
    tline = fgetl(fid) ;
    Dim = sscanf(tline, '%d %d %d') ;

    dataRoot = fgetl(fid) ;
    if ~java.io.File(dataRoot).isAbsolute()
        parent   = char(java.io.File(ListFN).getParentFile().getAbsolutePath());
        dataRoot = char(java.io.File([parent filesep dataRoot]).getCanonicalPath());
    end

    while 1,
        tline = fgetl(fid) ;
        if ~(strcmp('',tline))
            break ;
        end
    end

    fmtstr = [] ;
    for ii=1:numTissues
      fmtstr = [fmtstr '%s '] ;
    end
    fmtstr = [fmtstr '%d'] ; 

    labels = [] ;
    imgList = cell(1,numTissues) ;
    ii = 1 ;
    while 1
        tmpList = textscan(tline,fmtstr) ;
        if isempty(tmpList{end-1}) || all(isstrprop(tmpList{end-1}{1:end}, 'digit'))
            error(['Failed to parse line ' num2str(ii) ' following header of image list file ' imgList ' !!! Make sure that images for ' numTissues ' channel(s) are provided.']) ;
        end
        l = tmpList{end} ;
        for jj=1:numTissues
            if java.io.File(tmpList{jj}{1}).isAbsolute()
                imgList{ii,jj} = tmpList{jj}{1} ;
            else
                imgList{ii,jj} = char(java.io.File(dataRoot, tmpList{jj}{1}).getAbsolutePath()) ;
            end
        end
        labels = [labels; l] ;
        tline = fgetl(fid) ;
        ii = ii + 1 ;
        if ~ischar(tline), break, end
    end
    fclose(fid) ;
end

% Copyright (c) 2011, 2012 University of Pennsylvania. All rights reserved.
% See http://www.rad.upenn.edu/sbia/software/license.html or COPYING file.
%
% Contact: SBIA Group <sbia-software at uphs.upenn.edu>
