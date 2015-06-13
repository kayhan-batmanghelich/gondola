% This function is to use my solver for basis vectors
% In fact this function solves the following optimization problem:
%               min_{B}     D(V;BC) + l(y;X,B,w) 
%               s.t.        B => 0, C => 0
%                           B >= 1
%                           1^t B_i  <= lambda_const
%
%       where:
%               D(V;BC)   :    lambda_gen*\sum_m || V(m) - B*C(m) ||_{F}^{2}
%               l(y;X,B,w):    \sum_{k=1}^{K}  \lambda_disc*\eta_k *\sum_{i \in A_k}{max{ 0, \sum_m  1_{y_i=k}  < w(m,k) , B^t X_i(m) >  }^2} + \lambda_disc*\sum_{i \notin A_k}{max{ 0, \sum_m  -1_{y_i~=k}  < w(m,k) , B^t X_i(m) >  }^2} 
%
%       in which A_k are index sets of the k'th class 
%       V and C are tensors and superindex m, enumerates 
%       modalities.  K is number of classes and M is number of modalities.
%       \eta_k is the weight for the k'th class. w(m,k) is classifier
%       parameters for the m'th modality and k'th class. X_i(m) is the
%       m'th modality for the i'th subject.
function [B,Report]  = BSolver_MultiViewXY(B0,V,C,w,y,options)
    % decomposing options and creating necessary variables
    D = size(B0,1) ;
    r = size(B0,2) ;
    lambda_const = options.lambda_const ;
    BBMethod = options.BSolver_opt.BBMethod ;
    lb = zeros(D,1) ;
    ub = ones(D,1) ;
    % solving the problem using BBSolver
    f = @(x)obj_fun(x,V,C,w,y,options) ;
    g = @(x)obj_gradient(x,V,C,w,y,options) ;
    projFcn = @(x)ProjectionBasisVectors(x,lb, ub,lambda_const,D,r) ;
    x0  = projFcn(B0(:)) ;
    sol = gondola.spg(x0, f, g, BBMethod, projFcn, lb, ub) ;
    B = reshape(sol.par,D,r) ;
    % making the Report varibale and leaving the function 
    Report = sol ;
    Report = rmfield(Report,'par') ;
end


% objective function
function f = obj_fun(B,V,C,w,y,options)
    lambda_gen = options.lambda_gen ;
    lambda_disc = options.lambda_disc ;

    r = options.r ;
    D = options.D ;
    numChannels = options.numChannels  ;
    numClasses = options.numClasses ;
    class_N = options.class_N ;
    classWeight = options.classWeight ;
    nullWeight = options.nullWeight ;
    NLabel = sum(class_N) ;
    B = reshape(B,D,r) ;

    terms = zeros(2,1) ;
    
    % computing the generative term  
    stIdx = 1 ;
    endIdx = r ;
    for chanCnt=1:numChannels
          terms(1) = terms(1) + lambda_gen*norm((V(:,:,chanCnt) - B*C(:,:,chanCnt)),'fro')^2 ; 
          stIdx = stIdx + r  ;
          endIdx = endIdx + r ;
    end
    
    % computing the discriminative term
    if (numClasses > 2)
      for classCnt=1:numClasses
        stIdx = 1 ;
        endIdx = r ;
        decision1 = zeros(class_N(classCnt),1) ;
        decision2 = zeros(NLabel - class_N(classCnt),1) ;
        for chanCnt=1:numChannels
            V1 = V(:,y==classCnt,chanCnt) ;   % class "classCnt"
            V2 = V(:,y~=classCnt,chanCnt) ;   % whoever not in that class
            decision1 = decision1 + (V1'*( B*w(stIdx:endIdx,classCnt) )) ;
            decision2 = decision2 + (V2'*( B*w(stIdx:endIdx,classCnt) )) ;
            stIdx = stIdx + r  ;
            endIdx = endIdx + r ;
        end
        terms(2) = terms(2) + ...
                 lambda_disc*classWeight(classCnt)*sum( max(0,1 - (1).*decision1 + w(end,classCnt)).^2 ) + ...            % weight is w_i*lambda_disc
                 lambda_disc*nullWeight*sum( max(0,1 - (-1).*decision2 + w(end,classCnt)).^2 ) ;                           % weight is lambda_disc
      end
    else   % if there is only two classes, then we need to learn only one one model
        stIdx = 1 ;
        endIdx = r ;
        if ~isempty(class_N)   % make sure that there is any labeled data
          decision1 = zeros(class_N(1),1) ;
          decision2 = zeros(class_N(2),1) ;
          for chanCnt=1:numChannels
            V1 = V(:,y==1,chanCnt) ;   % class "classCnt"
            V2 = V(:,y~=1,chanCnt) ;   % whoever not in that class
            decision1 = decision1 + (V1'*( B*w(stIdx:endIdx) )) ;
            decision2 = decision2 + (V2'*( B*w(stIdx:endIdx) )) ;
            stIdx = stIdx + r  ;
            endIdx = endIdx + r ;
          end
          terms(2) = terms(2) + ...
                 lambda_disc*classWeight(1)*sum( max(0,1 - (1).*decision1 + w(end)).^2 ) + ...            % weight is w_i*lambda_disc
                 lambda_disc*classWeight(2)*sum( max(0,1 - (-1).*decision2 + w(end)).^2 ) ;                           % weight is lambda_disc
        else
          terms(2) = 0 ;
        end
    end    
    f = sum(terms) ;

end

% gradient function
function g = obj_gradient(B,V,C,w,y,options)
    lambda_gen = options.lambda_gen ;
    lambda_disc = options.lambda_disc ;

    r = options.r ;
    D = options.D ;
    numChannels = options.numChannels  ;
    numClasses = options.numClasses ;
    class_N = options.class_N ;
    classWeight = options.classWeight ;
    nullWeight = options.nullWeight ;
    NLabel = sum(class_N) ;
    B = reshape(B,D,r) ;


    % computing gradient of the generative term 
    g_rec = zeros(D,r) ; 
    stIdx = 1 ;
    endIdx = r ;
    %fprintf('computing g_rec: Time ') ;tic
    for chanCnt=1:numChannels
          g_rec = g_rec - 2*lambda_gen*( V(:,:,chanCnt)*C(:,:,chanCnt)' - B*(C(:,:,chanCnt)*C(:,:,chanCnt)') ) ; 
          stIdx = stIdx + r  ;
          endIdx = endIdx + r ;
    end
    %fprintf('[%f] \n ', toc) ;

    % computing gradient of the discriminative term
    g_disc1 = zeros(D,r) ;
    g_disc2 = zeros(D,r) ;
    if (numClasses > 2)
      for classCnt=1:numClasses
        stIdx = 1 ;
        endIdx = r ;
        decision1 = zeros(class_N(classCnt),1) ;
        decision2 = zeros(NLabel - class_N(classCnt),1) ;
        for chanCnt=1:numChannels
            V1 = V(:,y==classCnt,chanCnt) ;   % class "classCnt"
            V2 = V(:,y~=classCnt,chanCnt) ;   % whoever not in that class
            decision1 = decision1 + (V1'*( B*w(stIdx:endIdx,classCnt) )) ;
            decision2 = decision2 + (V2'*( B*w(stIdx:endIdx,classCnt) )) ;
            stIdx = stIdx + r  ;
            endIdx = endIdx + r ;
        end
        cost1 = 1 - (1).*decision1 + w(end,classCnt) ;
        cost2 = 1 - (-1).*decision2 + w(end,classCnt) ;
        index1 = find(cost1 > 0) ;
        index2 = find(cost2 > 0) ;
        stIdx = 1 ;
        endIdx = r ;
        for chanCnt=1:numChannels
            V1 = V(:,y==classCnt,chanCnt) ;   % class "classCnt"
            V2 = V(:,y~=classCnt,chanCnt) ;   % whoever not in that class
            if isempty(index1)
                g_disc1 = g_disc1 + zeros(size(g_rec)) ;
            else
                g_disc1 = g_disc1 + 2*lambda_disc*classWeight(classCnt)*classWeight(classCnt)*( -(+1)*V1(:,index1) )*cost1(index1)*w(stIdx:endIdx,classCnt)' ;
            end
            if isempty(index2)
                g_disc2 = g_disc2 + zeros(size(g_rec)) ;
            else
                g_disc2 = g_disc2 + 2*lambda_disc*nullWeight*( -(-1)*V2(:,index2) )*cost2(index2)*w(stIdx:endIdx,classCnt)' ;
            end
            stIdx = stIdx + r ;
            endIdx = endIdx + r ;
        end
      end
    else   % if there is only two classes, then we need to learn only one one model
        stIdx = 1 ;
        endIdx = r ;
        if ~isempty(class_N)   % make sure that there is any labeled data
          decision1 = zeros(class_N(1),1) ;
          decision2 = zeros(class_N(2),1) ;
          for chanCnt=1:numChannels
            V1 = V(:,y==1,chanCnt) ;   % class "classCnt"
            V2 = V(:,y~=1,chanCnt) ;   % whoever not in that class
            decision1 = decision1 + (V1'*( B*w(stIdx:endIdx) )) ;
            decision2 = decision2 + (V2'*( B*w(stIdx:endIdx) )) ;
            stIdx = stIdx + r  ;
            endIdx = endIdx + r ;
          end
          cost1 = 1 - (1).*decision1 + w(end) ;
          cost2 = 1 - (-1).*decision2 + w(end) ;
          index1 = find(cost1 > 0) ;
          index2 = find(cost2 > 0) ;
          stIdx = 1 ;
          endIdx = r ;
          for chanCnt=1:numChannels
            V1 = V(:,y==1,chanCnt) ;   % class "classCnt"
            V2 = V(:,y~=1,chanCnt) ;   % whoever not in that class
            if isempty(index1)
                g_disc1 = g_disc1 + zeros(size(g_rec)) ;
            else
                g_disc1 = g_disc1 + 2*lambda_disc*classWeight(1)*( -(+1)*V1(:,index1) )*cost1(index1)*w(stIdx:endIdx)' ;
            end
            if isempty(index2)
                g_disc2 = g_disc2 + zeros(size(g_rec)) ;
            else
                g_disc2 = g_disc2 + 2*lambda_disc*classWeight(2)*( -(-1)*V2(:,index2) )*cost2(index2)*w(stIdx:endIdx)' ;
            end
            stIdx = stIdx + r ;
            endIdx = endIdx + r ;
          end
        else
          g_disc1 = zeros(size(g_rec)) ;
          g_disc2 = zeros(size(g_rec)) ;
        end
    end
    g = g_rec + g_disc1 + g_disc2  ;
    g = g(:) ;
end

function y = phi(x)
 y = x.*x ;
end


function dy = dphi(x)
    dy = 2*x ;
end

function d2y = d2phi(x)
    d2y = 2 ;
end

% projection function
function Xp = ProjectionBasisVectors(x,lb, ub,lambda,D,r)
    B = reshape(x,D,r) ;
    Xp = zeros(D,r) ;

    for ii=1:r
        Xp(:,ii) = gondola.ProjectionOnUnitBoxSimplex(B(:,ii),lambda) ;
    end

    Xp = Xp(:) ;
end

% Copyright (c) 2011, 2012 University of Pennsylvania. All rights reserved.
% See http://www.rad.upenn.edu/sbia/software/license.html or COPYING file.
%
% Contact: SBIA Group <sbia-software at uphs.upenn.edu>
