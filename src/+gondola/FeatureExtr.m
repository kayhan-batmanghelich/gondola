% This function extracts features based on given basis vector and options
function FeatureExtr(IDs,V,y,W,opt) 
    arffFile = opt.ArffFileName ;
    mode = opt.ProjectionMode ;
    ID_Flag = opt.IsID_provided ;
    ClassName = opt.ClassName ;
    
    V = V(:,y~=0,:) ;       % do not extract features from unlabeled samples
    IDs = IDs(find(y~=0)) ;    % remove ids corresponding to unlabeled samples
    y = y(y~=0) ;
    FMatrix = ProjectionOnBasis(double(V),W,mode) ;
    
    for cnt=1:size(FMatrix,1)
	FLabel{cnt} = ['W' num2str(cnt)] ;
    end

    if ~(ID_Flag)		% we don't care about IDs
        gondola.Mat2Weka_Feature(FMatrix',FLabel,ClassName,y(:),arffFile,'Classifier') ;
    else
        gondola.Mat2Weka_Feature(FMatrix',FLabel,ClassName,y(:),arffFile,'Classifier',IDs) ;
    end
    
end

% This function does the projection
function Features = ProjectionOnBasis(V,W,Mode) 
    switch lower(Mode)
        case 'inner_product'	%	B^T X
                Features = [] ;
                if (ndims(W)==2)
                    for ii=1:size(V,3)
                        Features = [Features; W'*V(:,:,ii)] ;
                    end
                elseif (ndims(W)==3)
                    for ii=1:size(V,3)
                        Features = [Features; W(:,:,ii)'*V(:,:,ii)] ;
                    end
                else
                  error('dimensionality higher than 3 is not supported !!!!') ;
                end
        case 'projection'	%	(B^T B)^{-1} B^{T} X
                Features = [] ;
                if (ndims(W)==2)
                    for ii=1:size(V,3)
                        Features = [Features; inv(W'*W)*W'*V(:,:,ii)] ;
                    end
                elseif (ndims(W)==3)
                    for ii=1:size(V,3)
                        Features = [Features; inv(W(:,:,ii)'*W(:,:,ii))*W(:,:,ii)'*V(:,:,ii)] ;
                    end
                else
                  error('dimensionality higher than 3 is not supported !!!!') ;
                end

        case 'positive_projection'  %	min_c  || Bc - X ||, c>=0
                    Features = [] ;
                    if (ndims(W)==2)
                      for ii=1:size(V,3)
                         F = [] ;
                         for jj=1:size(V,2)
		             fprintf('non-negative projection on %d sample ... \n',jj) ;
                             x = lsqnonneg(W,V(:,jj,ii)) ;
                             F = [F x] ;
                         end
                         Features = [Features; F] ;
                      end
                    elseif (ndims(W)==3)
                      for ii=1:size(V,3)
                          F = [] ;
                          for jj=1:size(V,2)
		              fprintf('non-negative projection on %d sample ... \n',jj) ;
                              x = lsqnonneg(W(:,:,ii),V(:,jj,ii)) ;
                              F = [F x] ;
                          end
                          Features = [Features; F] ;
                      end
                    else
                      error('dimensionality higher than 3 is not supported !!!!!') ;
                    end
	case 'pca_projection'     % subtract the mean and then project
                Features = [] ;
                if (ndims(W)==2)
                    for ii=1:size(V,3)
                        Features = [Features ; W'*(V(:,:,ii) - repmat(mean(V(:,:,ii),2),1,size(V,2)))] ;
                    end
                elseif (ndims(W)==3)
                    for ii=1:size(V,3)
                        Features = [Features ; W(:,:,ii)'*(V(:,:,ii) - repmat(mean(V(:,:,ii),2),1,size(V,2)))] ;
                    end
                else
                   error('dimensionality higher than 3 is not supported !!!!!') ;
                end
        otherwise
            error('adniProjectData: this mode is not supported') ;
    end
end

% Copyright (c) 2011, 2012 University of Pennsylvania. All rights reserved.
% See http://www.rad.upenn.edu/sbia/software/license.html or COPYING file.
%
% Contact: SBIA Group <sbia-software at uphs.upenn.edu>
