% This function normalizes data according to the options. If the optiond
% has "nrm_params" as a field, it uses that otherwise it normalize it
% according to the mode
function [V, nrm_params] = normalizeData(V,opt)
    if isfield(opt,'nrm_params')
        nrm_params = opt.nrm_params ;
    else
        nrm_params = [] ;
    end
    switch opt.nrm_mode,
        case 0,
            % do nothing
        case 1,  % z-score
            if isfield(opt,'nrm_params')
                V_ave = nrm_params.V_ave  ;
                V_std = nrm_params.V_std  ;
            else
                for ii=1:size(V,3)
                  V_ave(:,:,ii) = mean(V(:,:,ii),2) ;
                  V_std(:,:,ii) = std(V(:,:,ii),0,2) ;
                  V_std(V_std==0) = 1 ;
                end
                nrm_params.V_ave = V_ave ;
                nrm_params.V_std = V_std ;
            end
            for ii=1:size(V,3)
                V(:,:,ii) = V(:,:,ii) - repmat(V_ave(:,:,ii),1,size(V(:,:,ii),2)) ;
                V(:,:,ii) = V(:,:,ii)./repmat(V_std(:,:,ii),1,size(V(:,:,ii),2)) ; 
            end
        case 2,  % between [0,1]
            if isfield(opt,'nrm_params')
                v_min = nrm_params.v_min ;
                v_max = nrm_params.v_max ;
                v_scale = nrm_params.v_scale ;
            else
                for ii=1:size(V,3)
                  v_min(:,:,ii) = min(V(:,:,ii),[],2) ;
                  v_max(:,:,ii) = max(V(:,:,ii),[],2) ;
                  v_scale(:,:,ii) = v_max(:,:,ii) - v_min(:,:,ii) ;
                  v_scale(v_scale==0) = 1 ;
                end
                nrm_params.v_min = v_min ;
                nrm_params.v_max = v_max ;
                nrm_params.v_scale = v_scale ;
            end
            for ii=1:size(V,3)
                V(:,:,ii) = V(:,:,ii) - repmat(v_min(:,:,ii),1,size(V(:,:,ii),2)) ;
                V(:,:,ii) = V(:,:,ii)./ repmat(v_scale(:,:,ii),1,size(V(:,:,ii),2)) ;
            end
        case 3,
            if isfield(opt,'nrm_params')
                sc = nrm_params.sc ;
            else
                for ii=1:size(V,3)
                  sc(ii) = min(max(V(:,:,ii))) ;
                end
                nrm_params.sc = sc ;
            end
            for ii=1:size(V,3)
              V(:,:,ii) = V(:,:,ii)/sc(ii)*opt.nrm_scale ;
            end
        otherwise,
            error('unknown normalization mode !') ;         
    end
end

% Copyright (c) 2011, 2012 University of Pennsylvania. All rights reserved.
% See http://www.rad.upenn.edu/sbia/software/license.html or COPYING file.
%
% Contact: SBIA Group <sbia-software at uphs.upenn.edu>
