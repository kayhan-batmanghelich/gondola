% this function decides what to with the file based on file extension
function [img,hdr] = readimage(fn)
    try
        abs_fn = char(java.io.File(fn).getAbsolutePath()) ;
        [img, origin, spacing] = gondola.readmedicalimage(abs_fn) ;
        hdr.PixelDimensions = spacing ;
        hdr.Dimensions = size(img) ;
        hdr.origin = origin ;
    catch exception
        disp(exception.message) ;
        fprintf(['Failed to read input image file:  ' fn '\n']) ;
    end
end

% Copyright (c) 2011, 2012 University of Pennsylvania. All rights reserved.
% See http://www.rad.upenn.edu/sbia/software/license.html or COPYING file.
%
% Contact: SBIA Group <sbia-software at uphs.upenn.edu>
