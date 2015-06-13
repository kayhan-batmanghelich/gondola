%   Usage:
%	Mat2Weka_Feature(FMatrix,FLabel,ClassName,SampleLabel,Arff_FN,mode)
%	
%	FMatrix : Matrix holding features (m x n): m: numbwer of samples, n: number of features
%	FLabel	: Labels of features (name for each feature), this is a cell
%	ClassName : name of classes, this is a cell
%	SampleLabel : this is a columns vector holding sample labels
%	Arff_FN     : output file, it is better if it ends with .arff
%	mode:	     : either 'Classifier' or 'Regression'
%
% 	Example:
%		Fmatrix = rand(42,2)
%		Flabel = {'age','color'}
%		ClassName = {'1','2'}
%		SampleLabel = double(randn(42,1)>0) + 1
%		Arff_FN =  'test.arff'
% 		mode = 'Classifier'
%
%		Mat2Weka_Feature(Fmatrix,Flabel,ClassName,SampleLabel,Arff_FN,mode)
%		
%		Note: If IDs are also provided as the last argument to the fucntion, the last attributes would be ID and one before that would be class
%			label. Obviously, we should have as many ID as number of samples.

function Mat2Weka_Feature(FMatrix,FLabel,ClassName,SampleLabel,Arff_FN,mode,IDs)

%'@RELATION rigein_ID\n@ATTRIBUTE GMAXP REAL\n@ATTRIBUTE FNULL REAL\n@ATTRIBUTE HSVPP REAL\n@ATTRIBUTE GMAXP REAL\n@ATTRIBUTE FNULL REAL\n@ATTRIBUTE HSVPP REAL\n

if (nargin<6) 
	error('not enough input arguments!') ;
end

if ((nargin==7) && (length(SampleLabel)~=length(IDs)))
	error('number of samples and numbe of IDs should be the same !!!') ;
end

Header1 = [] ;
Header2 = [] ;
Header3 = [] ;
precisionFormat = '%.4f';  %'%2.3f' ;


Header1 = ['@RELATION Class_ID\n'] ;
for Cnt=1:length(FLabel)
    Header1 = [Header1 '@ATTRIBUTE ' FLabel{Cnt} ' REAL\n'] ;
end

switch mode
    case 'Classifier'
        
        %@ATTRIBUTE class {1,2,3,4,5,7,9,10}\n
        Header1 = [Header1 '@ATTRIBUTE class {' ] ;
        for Cnt=1:length(ClassName)
            Header1 = [Header1  ClassName{Cnt} ',' ] ;
        end
        
    case 'Regression'
        Header1 = [Header1 '@ATTRIBUTE class REAL' ] ;
    otherwise
        error('Unkown mode for Mat2Weka Function !!!') ;
end

Header1 = [Header1(1:end-1) '}\n'] ;

if (nargin==7)  % add id as one extra attribute
	Header1 = [Header1 '@ATTRIBUTE ' 'IDs' ' STRING\n'] ;
end

% %@ATTRIBUTE class {1,2,3,4,5,7,9,10}\n
% Header1 = [Header1 '@ATTRIBUTE class {' ] ;
% for Cnt=1:length(ClassName)
%     Header1 = [Header1  ClassName{Cnt} ',' ] ;
% end


Header2 = ['@DATA\n'] ;

for Cnt=1:length(FLabel)
    Header3 = [Header3 precisionFormat ','] ;
end

switch mode
    case 'Classifier'
	if (nargin==6)  
		Header3 = [Header3 '%i\n'] ;          
	elseif (nargin==7)
		Header3 = [Header3  '%i,%s\n'] ;          % add ID as the last attribute
	end
    case 'Regression'
	if (nargin==6)
		Header3 = [Header3 precisionFormat '\n'] ;
	else
		Header3 = [Header3 precisionFormat ',%s\n'] ;          % add ID as the last attribute
	end
    otherwise
        error('Unkown mode for Mat2Weka Function !!!') ;
end
%Header3 = [Header3 '%i\n'] ;

Output = [FMatrix  double(SampleLabel)] ;                        % size (Output) = #Samples x # Features

fin = fopen(Arff_FN,'w+');
fprintf(fin,Header1) ;
fprintf(fin,Header2) ;
if (nargin==6)  
	fprintf(fin,Header3,Output') ;
elseif (nargin==7)
	Output = Output' ;
	for cnt=1:size(Output,2)
		inst = Output(:,cnt) ;
		fprintf(fin,Header3 ,inst,IDs{cnt}) ;		
	end
end 
fclose(fin);

% Copyright (c) 2011, 2012 University of Pennsylvania. All rights reserved.
% See http://www.rad.upenn.edu/sbia/software/license.html or COPYING file.
%
% Contact: SBIA Group <sbia-software at uphs.upenn.edu>
