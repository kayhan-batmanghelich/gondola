% this version of solver solve the following objective:
%               min     D(V;BC) + l(y;X,B,w) + 0.5*|| w ||_{2}^{2}
%               s.t.    B => 0, C => 0
%                       B >= 1
%                       1^t B_i  <= lambda_const
%
%       where:
%               D(V;BC)   :    lambda_gen*|| V - BC||_{F}^{2}
%               l(y;X,B,w):    \sum_{k=1}^{K}  \lambda_disc*\eta_k *\sum_{i \in A_k}{max{ 0, \sum_m  1_{y_i=k}  < w(m,k) , B^t X_i(m) >  }^2} + \lambda_disc*\sum_{i \notin A_k}{max{ 0, \sum_m  -1_{y_i~=k}  < w(m,k) , B^t X_i(m) >  }^2} 
%
%       in which A_k are index sets of the k'th class 
%       V, B and C are tensors and superindex m, enumerates 
%       modalities.  K is number of classes and M is number of modalities.
%       \eta_k is the weight for the k'th class. w(m,k) is classifier
%       parameters for the m'th modality and k'th class. X_i(m) is the
%       m'th modality for the i'th subject. 
function [B,C,w,Report] = MultiViewX(Data,ConstVars) 
   
    % check if Data and ConstVars are consistent with requirement of the algorithm
    checkOptions(Data,ConstVars) ; 
    % Initialize if intial variables are not initialized. Notice that if initial blocks exist, it means the algorithm should continiue from the latest iteration saved in a the result file
    numChannels = ConstVars.numChannels  ;
    numClasses = ConstVars.numClasses ;
    N = ConstVars.N ;
    r = ConstVars.r ;
    D = ConstVars.D ;
    randSeed = ConstVars.randSeed ;
    rand('seed',randSeed)
    randn('seed',randSeed)
    if ~(isfield(Data,'W0') && isfield(Data,'H0') && isfield(Data,'w0') )
        display('Randomg initialization of blocks (w,B,C) .... ') ; 
        Data.W0 = rand(D,r,numChannels) ;
        for chanCnt=1:numChannels
             Data.W0(:,:,chanCnt) = Data.W0(:,:,chanCnt)./repmat(sum(Data.W0(:,:,chanCnt)),D,1)*ConstVars.lambda_const/2 ;   % to make initialization feasible
        end
        Data.H0 = rand(r,N,numChannels) ;
        if (numClasses > 2)
            Data.w0 = randn(r*numChannels,numClasses) ;
        else
            Data.w0 = randn(r*numChannels,1) ;
        end
    end

    % disassemble constant variables
    lambda_gen = ConstVars.lambda_gen ;
    lambda_disc = ConstVars.lambda_disc ;
    lambda_const = ConstVars.lambda_const ;
    lambda_stab = ConstVars.lambda_stab ;
    MAXITR = ConstVars.MAXITR ;
    ZSBT   = ConstVars.ZSBT ;
    tol = ConstVars.tol ;
    saveAfterEachIteration = ConstVars.saveAfterEachIteration ;
    DataFile = ConstVars.DataFile ;
    logFn = [DataFile(1:end-4)  '.log'] ;
    classWeight = ConstVars.classWeight ;
    
    Monitor_Bsol = ConstVars.Monitor_Bsol  ;
    Monitor_Csol = ConstVars.Monitor_Csol  ;

    % disassemble data
    V = Data.V ;
    y = Data.y ;
    B0 = Data.W0 ;
    C0 = Data.H0 ;
    w0  = Data.w0 ;
    if isfield(Data,'iter0')
        iter0 = Data.iter0 ;
    else
        iter0 = 1 ;
    end
    
    Vtmp = [] ;    % matrix holding all data
    class_N = [] ;
    ytmp = [] ;
    svm_opt = ['-s 2   -e 0.0001   -B 1   -c ' num2str(lambda_disc)  ' '] ;
    for classCnt=0:numClasses
        Vtmp =  [ Vtmp   V(:,y==classCnt,:) ] ;     
        ytmp =  [ ytmp; classCnt*ones(sum(y==classCnt),1) ] ;
        if (classCnt==0)
            % do nothing!
        else
            class_N = [class_N; sum(y==classCnt) ] ;
            svm_opt = [svm_opt  ' -w' num2str(classCnt) ' ' num2str(classWeight(classCnt)) '  ' ] ;
        end
    end
    y = ytmp ;
    V = Vtmp ;
    clear ytmp  Vtmp
    if (sum(lambda_disc)==0)
        if (numClasses > 2)  % multi-class
            w = zeros(r*numChannels+1,numClasses) ;
        else   
            w = zeros(r+1,1) ;
        end
    else
        w = w0 ;
    end
    ConstVars.class_N = class_N ;


    % Perform alternating minimization
    B =  B0 ;
    C =  C0 ;
    Terms_Hist = [] ;
    Obj_Hist = [] ;
    [Obj_Hist, Terms_Hist] = UpdateHistory(Obj_Hist,Terms_Hist,V,B,C,w,y,class_N,ConstVars) ;
    tic
    for iter = iter0:MAXITR
        % my BBsolver for B
        if mod(iter,3) == 1    % optimization wrt to B
            blockName = 'B' ;
            B_old = B ;
            B0 = B ;
            [B,Report]  = gondola.BSolver_MultiViewY(B0,V,C,w,y,ConstVars) ;
            [Obj_Hist, Terms_Hist] = UpdateHistory(Obj_Hist,Terms_Hist,V,B,C,w,y,class_N,ConstVars) ;
            if ((Obj_Hist(end) > Obj_Hist(end-1)) && Monitor_Bsol)
                    B = B_old ;
                    Terms_Hist(:,end) = [] ;
                    Obj_Hist(end) = [] ;
            end
        elseif mod(iter,3) == 2    % optimization wrt to w
            blockName = 'w' ;
            if (lambda_disc~=0) 
                yl = y(y~=0) ;   % labeled data
                fMat = [] ;     % storing feature matrix
                for chanCnt=1:numChannels
                    fMat = [fMat  V(:,y~=0,chanCnt)'*B(:,:,chanCnt)] ;
                end
                fMat = sparse(fMat) ;
                model = train(yl,fMat,svm_opt) ;
                w = model.w' ;
            end
            [Obj_Hist, Terms_Hist] = UpdateHistory(Obj_Hist,Terms_Hist,V,B,C,w,y,class_N,ConstVars) ;
        else   % optimization wrt to C
            blockName = 'C' ;
            C_old = C ;
            C0 = C ;
            for chanCnt=1:numChannels
                if strcmpi(ConstVars.csolver, 'mosek')
                    [C(:,:,chanCnt),Report]  = gondola.CSolver_mosek(C0(:,:,chanCnt),V(:,:,chanCnt),B(:,:,chanCnt),w,y,ConstVars) ;
                else
                    [C(:,:,chanCnt),Report]  = gondola.CSolver_spg(C0(:,:,chanCnt),V(:,:,chanCnt),B(:,:,chanCnt),w,y,ConstVars) ;
                end
                [Obj_Hist, Terms_Hist] = UpdateHistory(Obj_Hist,Terms_Hist,V,B,C,w,y,class_N,ConstVars) ;
                if ((Obj_Hist(end) > Obj_Hist(end-1)) && Monitor_Csol)
                    C(:,:,chanCnt) = C_old(:,:,chanCnt) ;
                  Terms_Hist(:,end) = [] ;
                  Obj_Hist(end) = [] ;
                end
            end
        end
        %mosekend
        curTime = datestr(now) ;
        fprintf(1,'(%s)-Iteration(%s) %d -- obj: %g, D(.;.): %g, l(.;.): %g, ||w||:%g, ||C||:%g \n',curTime,blockName,iter, Obj_Hist(end),...
                                                Terms_Hist(1,end), Terms_Hist(2,end), Terms_Hist(3,end), Terms_Hist(4,end));

        logFid = fopen(logFn,'at+') ;
        fprintf(logFid,'(%s)-Iteration(%s) %d -- obj: %g, D(.;.): %g, l(.;.): %g, ||w||:%g, ||C||:%g \n',curTime,blockName,iter, Obj_Hist(end),...
                                                Terms_Hist(1,end), Terms_Hist(2,end), Terms_Hist(3,end), Terms_Hist(4,end));

        fclose(logFid) ;
        Report.Obj_Hist     = Obj_Hist ;
        Report.Terms_Hist   = Terms_Hist ;
        if (saveAfterEachIteration)
            %if ismember(iter,[1:10 20:20:100 100:50:300 300:5000:10000])   % save less often as it goes
            elapsedTime = toc ;
            if (elapsedTime > 7200)
                display('saving intermediate results ...') ;
                logFid = fopen(logFn,'at+') ;
                fprintf(logFid,'saving intermediate results ...') ;
                if exist(ConstVars.DataFile,'file')
                    save(DataFile,'-append','w','B','C','Report','iter') ;
                else
                    save(DataFile,'w','B','C','Report','iter') ;
                end
                fprintf(logFid,'Done! \n')  ;
                fclose(logFid) ;
                tic
            end
        end
    end
    
    Report.Obj_Hist     = Obj_Hist ;
    Report.Terms_Hist   = Terms_Hist ;
    
    svm_opt = [svm_opt ' -v 10 '] ;
    fMat = [] ;     % storing feature matrix
    for chanCnt=1:numChannels
          fMat = [fMat  V(:,y~=0,chanCnt)'*B(:,:,chanCnt)] ;
    end
    fMat = sparse(fMat) ;
    Report.Accuracy = train(y(y~=0),fMat,svm_opt) ;
    if (saveAfterEachIteration)
	    %if exist(ConstVars.DataFile,'file')
	        %save(DataFile,'-append','w','B','C','Report','iter','ConstVars') ;
	    %end
            save(DataFile,'w','B','C','Report','iter','ConstVars') ;
            logFid = fopen(logFn,'at+') ;
            fprintf(logFid,'Learning process is done ! \n') ;
            fclose(logFid) ;
    end
end


% update objective history and other reports
function [Obj_Hist, Terms_Hist] = UpdateHistory(Obj_Hist,Terms_Hist,V,B,C,w,y,class_N,ConstVars) 
    lambda_gen = ConstVars.lambda_gen ;
    lambda_disc = ConstVars.lambda_disc ;
    lambda_stab = ConstVars.lambda_stab ;   
   
    r = size(B,2) ;
    numChannels = ConstVars.numChannels  ;
    numClasses = ConstVars.numClasses ;
    classWeight = ConstVars.classWeight ;
    nullWeight = ConstVars.nullWeight ;
    NLabel = sum(class_N) ;
    
    terms = zeros(4,1) ;
    
    % computing the generative term and the the stabilizer term 
    stIdx = 1 ;
    endIdx = r ;
    for chanCnt=1:numChannels
          terms(1) = terms(1) + lambda_gen*norm((V(:,:,chanCnt) - B(:,:,chanCnt)*C(:,:,chanCnt)),'fro')^2 ; 
          terms(4) = terms(4) + lambda_stab*sum(sum(C(:,:,chanCnt).*C(:,:,chanCnt))) ;
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
            decision1 = decision1 + (V1'*( B(:,:,chanCnt)*w(stIdx:endIdx,classCnt) )) ;
            decision2 = decision2 + (V2'*( B(:,:,chanCnt)*w(stIdx:endIdx,classCnt) )) ;
            stIdx = stIdx + r  ;
            endIdx = endIdx + r ;
        end
        terms(2) = terms(2) + ...
                 lambda_disc*classWeight(classCnt)*sum( max(0,1 - (1).*decision1 + w(end,classCnt)).^2 ) + ...            % weight is w_i*lambda_disc
                 lambda_disc*nullWeight*sum( max(0,1 - (-1).*decision2 + w(end,classCnt)).^2 ) ;                           % weight is lambda_disc
        terms(3) =  terms(3) + 0.5*w(:,classCnt)'*w(:,classCnt) ;    
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
            decision1 = decision1 + (V1'*( B(:,:,chanCnt)*w(stIdx:endIdx) )) ;
            decision2 = decision2 + (V2'*( B(:,:,chanCnt)*w(stIdx:endIdx) )) ;
            stIdx = stIdx + r  ;
            endIdx = endIdx + r ;
          end
          terms(2) = terms(2) + ...
                 lambda_disc*classWeight(1)*sum( max(0,1 - (1).*decision1 + w(end)).^2 ) + ...            % weight is w_i*lambda_disc
                 lambda_disc*classWeight(2)*sum( max(0,1 - (-1).*decision2 + w(end)).^2 ) ;                           % weight is lambda_disc
          terms(3) =  0.5*(w'*w) ;    
        else
          terms(2) = 0 ;
          terms(3) = 0 ;
        end
    end    
    obj_val = sum(terms) ;
    Terms_Hist = [Terms_Hist terms ] ;
    Obj_Hist(end + 1) = obj_val;
end


% this function is a sanity check. It makes sure that all required arguments and data are provided
function checkOptions(Data,ConstVars) 
      % check required fields
      requiredOptions = {'lambda_gen','lambda_disc','lambda_const','lambda_stab','N','r','D',...
                        'MAXITR','ZSBT','tol','BSolver_opt','saveAfterEachIteration',...
                        'DataFile','Monitor_Bsol','Monitor_Csol','numChannels'} ;
     configFieldsNames = fieldnames(ConstVars) ;
     for cnt=1:length(requiredOptions)
        if ~ismember(requiredOptions{cnt},configFieldsNames)
            error([ 'This option is necessary, make sure that the config has it : ' requiredOptions{cnt}]) ;
        end
     end

     % check required fields for Data
     requiredOptions = {'V','y'} ;
     configFieldsNames = fieldnames(Data) ;
     for cnt=1:length(requiredOptions)
        if ~ismember(requiredOptions{cnt},configFieldsNames)
            error([ 'This option is necessary, make sure that the config has it : ' requiredOptions{cnt}]) ;
        end
     end

     % check whether the provided Data is consistent with the needs of the data
     y = reshape(Data.y,[1 length(Data.y)]) ;
     if (min(unique(y))<0)
         error('Values of the labels can be at least 0 (for un-labeled data) ! Do not use negative values for labels !! ') ;
     end

     if ~(isequal(y,floor(y)))  
        error('labels must be integer, multi-class and regression have not been implemented yet !') ;
     end

     numChannels = ConstVars.numChannels ;
     if (numChannels~=size(Data.V,3))
        error('number of channels does not match with size of input tensor data !') ;
     end
end
    
% Copyright (c) 2011, 2012 University of Pennsylvania. All rights reserved.
% See http://www.rad.upenn.edu/sbia/software/license.html or COPYING file.
%
% Contact: SBIA Group <sbia-software at uphs.upenn.edu>
