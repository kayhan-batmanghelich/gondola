% Call this function with 'help' as argument to display the help, i.e., gondola.main('help').
function main(varargin)
    % parse function/program arguments
    Args = parseargs(varargin);
    if strcmp(Args.what, 'help')
        print_help();
        if isdeployed() && isempty(varargin)
            % only to have a none-zero exit code in case of the gondola program
            error('No arguments given! See help output for a list of required arguments.')
        end
        return
    end

    % execute requested subfunction
    switch Args.what

        % --------------------------------------------------------------------
        % learn basis vectors using selected algorithm
        case 'learn'

            % initialize data and constants
            [Data, ConstVars] = initialize(Args);
            % learn basis vectors
            switch lower(ConstVars.algo)
                %  XY input       : V is tensor
                %  Y  basis       : B is tensor
                %  C coefficients : tensor, in case of _freeC, these coefficients can also take negative values.
                case 'multiviewy',        gondola.MultiViewY       (Data, ConstVars);
                case 'multiviewxy',       gondola.MultiViewXY      (Data, ConstVars);
                case 'multiviewxy_freec', gondola.MultiViewXY_freeC(Data, ConstVars);
                otherwise
                    error(['Invalid algo: ' ConstVars.algo ' !!!']);
            end

        % --------------------------------------------------------------------
        % save basis vectors as images
        case 'show'

            gondola.writeBasisImages(Args.datafile, Args.imagelistfile, Args.outputdir);

        % --------------------------------------------------------------------
        % extract features based on learned basis vectors
        case 'extract'

            % initialize data and constants
            [Data, ConstVars] = initialize(Args);
            % load basis vectors
            B = [];
            load(Args.datafile, 'B');
            if isempty(B)
                error(['Failed to load basis vectors from ' Args.datafile ' !!!']);
            end
            % read IDs from list file
            IDs = textread(Args.idlistfile, '%s', 'delimiter', '\n');
            % extract features
            FtrExtrOptions.ArffFileName   = Args.featuresfile;
            FtrExtrOptions.ProjectionMode = ConstVars.ProjectionMode;
            FtrExtrOptions.IsID_provided  = true;    % I kept this only for historical reasons
            FtrExtrOptions.ClassName      = {};
            for classCnt=1:ConstVars.numClasses
                 FtrExtrOptions.ClassName{classCnt} = num2str(classCnt);
            end
            gondola.FeatureExtr(IDs, Data.V, Data.y, B, FtrExtrOptions);

        % --------------------------------------------------------------------
        otherwise
            error(['Invalid subfunction: ' Args.what '! Valid choices are "learn", "show", and "extract".']);

    end
end

% --------------------------------------------------------------------------------------
% Parse input arguments.
function Args = parseargs(argv)
    % show help if called without arguments
    if isempty(argv) || ismember(argv{1}, {'-h', '--help', 'help'})
        Args.what = 'help';
        return
    end

    % string to display in error message as hint on how the function help can be accessed
    if isdeployed()
        help   = '"gondola --help"';
        prefix = '--';
    else
        help   = '"gondola(''help'')"';
        prefix = '';
    end

    % ------------------------------------------------------------------------
    % collect arguments into structure
    Args = [];
    Args.what    = lower(argv{1});
    Args.verbose = 0;
    i = 2;
    while i <= size(argv, 2)
        % option name
        o = lower(argv{i});
        if ~ischar(o), error('Option names must be strings !!!'), end
        % '--' option name prefix as in --configfile is optional if called as MATLAB function
        if ~isdeployed() && (size(o, 2) < 2 || ~strcmp(o(1:2), '--'))
            o = ['--' o];
        end
        % parse option
        switch o
            case {'--imagelistfile', '--idlistfile', '--configfile', '--datafile',...
                  '--featuresfile',  '--outputdir',  '--randseed'}
                if i == size(argv)
                    error(['Option ' prefix o ' requires an argument !!! See ' help ' for details.']);
                end
                i = i + 1;
                Args.(o(3:size(o, 2))) = strtrim(argv{i});
            case '--continue'
                Args.continue = true;
            case {'-v', '--verbose'}
                Args.verbose = Args.verbose + 1;
            case {'-h', '--help'}
                Args.what = 'help';
                return
            otherwise
                error(['Invalid option: ' prefix o ' !!! See ' help ' for a list of available options.']);
        end
        % next option
        i = i + 1;
    end

    % ------------------------------------------------------------------------
    %  check arguments and that all required options were specified

    % what
    if ~ismember(Args.what, {'learn', 'show','extract'})
        error(['Invalid subfunction: ' Args.what ' !!! See ' help ' for a list of valid subfunctions.']);
    end

    % outputdir
    if ~isfield(Args, 'outputdir')
        Args.outputdir = pwd;
    end
    Args.outputdir = char(java.io.File(Args.outputdir).getAbsolutePath());
    if strcmp(Args.what, 'show') && ~exist(Args.outputdir, 'dir')
        [status, msg, ~] = mkdir(Args.outputdir);
        if status > 1
            error(['Failed to create output directory ''' Args.outputdir ''' !!! Error: ' msg]);
        end
    end

    % imagelistfile
    if ~isfield(Args, 'imagelistfile')
        error(['Missing image list file argument !!! See ' help ' for a list of required arguments.']);
    end
    Args.imagelistfile = char(java.io.File(Args.imagelistfile).getAbsolutePath());
    if ~exist(Args.imagelistfile, 'file')
        error(['Image list file ''' Args.imagelistfile ''' does not exist !!!']);
    end

    % idlistfile
    if strcmp(Args.what, 'extract')
        if ~isfield(Args, 'idlistfile')
            error(['Missing ID list file argument !!! See ' help ' for a list of required arguments.']);
        end
        Args.idlistfile = char(java.io.File(Args.idlistfile).getAbsolutePath());
        if ~exist(Args.idlistfile, 'file') 
            error(['ID list file ''' Args.idlistfile ''' does not exist !!!']);
        end
    elseif ~isfield(Args, 'idlistfile')
        Args.idlistfile = 'unused';
    end

    % configfile
    if strcmp(Args.what, 'learn')
        if ~isfield(Args, 'configfile')
            Args.configfile = 'gondola.cfg';
        end
        Args.configfile = char(java.io.File(Args.configfile).getAbsolutePath());
        if ~exist(Args.configfile, 'file') 
            error(['Configuration file ''' Args.configfile ''' does not exist !!!']);
        end
    elseif ~isfield(Args, 'configfile')
        Args.configfile = 'unused';
    end

    % datafile
    if ~isfield(Args, 'datafile') 
        Args.datafile = 'gondola.mat';
    end
    if strcmp(Args.what, 'learn')
        if ~java.io.File(Args.datafile).isAbsolute()
            Args.datafile = char(java.io.File(Args.outputdir, Args.datafile).getAbsolutePath());
        end
    else
        Args.datafile = char(java.io.File(Args.datafile).getAbsolutePath());
    end
    if strcmp(Args.what, 'learn')
        try
            [dir, ~, ext, ~] = fileparts(Args.datafile);
        catch exception
            error(['Failed to get file parts of datafile path ''' Args.datafile ''' !!! Error: ' exception.message]);
        end
        if ~ismember(ext, {'.mat', '.txt'})
            Args.datafile = [Args.datafile '.mat'];
        end
        if ~exist(dir, 'dir')
            [status, msg, ~] = mkdir(dir);
            if status > 1
                error(['Failed to create output directory ''' dir ''' !!! Error: ' msg]);
            end
        end
    elseif ~exist(Args.datafile, 'file')
        error(['Result file ''' Args.datafile ''' of basis vector learning does not exist !!!']);
    end

    % featuresfile
    if ~isfield(Args, 'featuresfile')
        Args.featuresfile = 'features.arff';
    end
    if ~java.io.File(Args.featuresfile).isAbsolute()
        Args.featuresfile = char(java.io.File(Args.outputdir, Args.featuresfile).getAbsolutePath());
    end
    try
        [dir, ~, ext, ~] = fileparts(Args.featuresfile);
    catch exception
        error(['Invalid argument for featuresfile option !!! Error: ' exception.message]);
    end
    if ~isequal(ext, '.arff')
        Args.featuresfile = [Args.featuresfile '.arff'];
    end
    if strcmp(Args.what, 'extract') && ~exist(dir, 'dir')
        [status, msg, ~] = mkdir(dir);
        if status > 1
            error(['Failed to create output directory ''' dir ''' !!! Error: ' msg]);
        end
    end

    % continue
    if ~isfield(Args, 'continue')
        Args.continue = false;
    elseif strcmp(Args.what, 'learn') && Args.continue
        try 
            load(Args.datafile, 'iter');
        catch exception
            disp(exception.message);
            Args.continue = false;
        end
        if ~exist('iter', 'var') || iter < 2
            Args.continue = false;
        end
        if ~Args.continue
            warning('Cannot continue learning process as previous results were either not found or are invalid. Starting from scratch instead.');
        end
    end

    % randseed - otherwise, set by configuration or initconfig()
    if isfield(Args, 'randseed')
        try
            Args.randseed = str2double(Args.randseed);
        catch exception
            error(['Invalid seed value for random number generator specified: ' Args.randseed]);
        end
    end

    %  print options for reference
    if Args.verbose > 0
        disp(Args);
    end
end

% --------------------------------------------------------------------------------------
% read configuration from file
function Config = readconfig(configfile)
    Config = [];
    % read configuration file content
    fid = fopen(configfile);
    if (fid == -1)
        error(['Failed to open configuration file: ' fileName]);
    end
    content = textscan(fid, '%s%s', 'delimiter', ':', 'commentStyle', '#');
    fclose(fid);
    % process lines and convert strings to proper type
    for lineno = 1:length(content{1})
        o = strtrim(content{1}{lineno});
        a = strtrim(content{2}{lineno});
        n = str2double(a);
        if ~isnan(n)
            Config.(o) = n;
        elseif strcmpi(a, 'true')
            Config.(o) = true;
        elseif strcmpi(a, 'false')
            Config.(o) = false;
        else
            Config.(o) = a;
        end
    end
end

% ----------------------------------------------------------------------------
% set default configuration for missing configuration options
function Config = initconfig(Config, Args)
    if ~isfield(Config, 'csolver')
        Config.csolver = 'SPG';
    end
    if isfield(Args, 'randseed')
        Config.randseed = Args.randseed;
    elseif ~isfield(Config, 'randseed')
        Config.randseed = 0;
    end
    if ~isfield(Config, 'nrm_scale')
        Config.nrm_scale = 5;
    end
    if ~isfield(Config, 'projectionmode')
        Config.projectionmode = 'inner_product';
    end
    if ~isfield(Config, 'numbatchbasisvectors')
        Config.numbatchbasisvectors = 2;
    end
    if ~isfield(Config, 'downsampleratio')
        Config.downsampleratio = 2;
    end
    % MOSEK available ?
    mosek = exist('mosekopt', 'file');
    if strcmpi(Config.csolver, 'mosek') && ~mosek
        error('Cannot use MOSEK to solve for C !!! Install MOSEK and rebuild GONDOLO with USE_MOSEK set to ON or use SPG as C solver instead.');
    end
    if ~ismember(lower(Config.csolver), {'spg', 'mosek'})
        error(['Invalid C solver option value: ' Config.csolver]);
    end
end

% ----------------------------------------------------------------------------
% initialize data and algorithm options given the input arguments and configuration
function [Data, ConstVars] = initialize(Args)
    global ZSBT; ZSBT = 1e-20; % zero substitute

    Data      = []; % image data and labels
    ConstVars = []; % algorithm settings

    % read data and initialize settings for learning
    if strcmp(Args.what, 'learn')
        % read configuration
        Config = readconfig(Args.configfile);
        Config = initconfig(Config, Args);
        % read data
        [Data.V, Data.y, Dim] = gondola.ReadDataFromCOMPAREConfigFile(Args.imagelistfile, Config.downsampleratio);
        if min(Data.y) < 0
            error('Minimum value for the lables can be zero for unlabeled data');
        end

        numClasses  = length(unique(Data.y)) - any(Data.y == 0);
        classN      = zeros(numClasses, 1);
        classWeight = ones (numClasses, 1);
        for classCnt = 1:numClasses
            classN(classCnt)      = length(find(Data.y == classCnt));
            classWeight(classCnt) = 1 / classN(classCnt);
        end

        ConstVars.classWeight            = classWeight;
        ConstVars.numClasses             = numClasses;
        ConstVars.nullWeight             = 1 / sum(classN);                 % this value is used to weight part
                                                                            % of the loss function that penalizes
                                                                            % not being class(i) in one_vs_all
                                                                            % multi-class scenario
        % normalize data
        [Data.V, nrm_params]             = gondola.normalizeData(Data.V, Config);
        ConstVars.nrm_params             = nrm_params;
        ConstVars.nrm_mode               = Config.nrm_mode;
        ConstVars.nrm_scale              = Config.nrm_scale;
        % general problem variables
        ConstVars.algo                   = Config.algo;
        ConstVars.csolver                = Config.csolver;
        ConstVars.DownSampleRatio        = Config.downsampleratio;
        ConstVars.randSeed               = Config.randseed;
        ConstVars.saveAfterEachIteration = Config.saveaftereachiteration;
        ConstVars.ProjectionMode         = Config.projectionmode;
        ConstVars.lambda_const           = prod(Dim) * Config.lambda_const;
        if isfield(Config, 'lambda_laplac')
            warning('You are using laplacian regularization of the method, be careful, this is not an official part of the package !!!') ;
            ConstVars.lambda_laplac      = Config.lambda_laplac;            % regularizer for the Laplacian term if used
        end
        ConstVars.r                      = Config.numbasisvectors;          % number of basis vectors
        ConstVars.N                      = length(Data.y);                  % number of training samples
        ConstVars.numChannels            = size(Data.V, 3);
        ConstVars.lambda_gen             = Config.lambda_gen / ConstVars.N; % weight for Frobinous norm
        ConstVars.lambda_disc            = Config.lambda_disc;              % weight for the second class loss function
        ConstVars.lambda_stab            = Config.lambda_stab;              % stabilizer regularizer for the coefficients
        ConstVars.MAXITR                 = Config.maxitr;                   % MAXITR for optimization
        ConstVars.ZSBT                   = ZSBT;                            % zero substituite
        ConstVars.tol                    = Config.tol;                      % tolerance for optimization (used 1e-3)
        ConstVars.numBatchBasisVector    = Config.numbatchbasisvectors;     % number of batch of basis vectors to be optimized together
        ConstVars.D1                     = Dim(1);                          % x: original image size
        ConstVars.D2                     = Dim(2);                          % y: original image size
        ConstVars.D3                     = Dim(3);                          % z: original image size
        ConstVars.D                      = prod(Dim);
        % options for the BSolver
        ConstVars.BSolver_opt.BBMethod   = Config.bbmethod;           
        % optimizer parameter
        ConstVars.Monitor_Bsol           = Config.monitor_bsol;             % this options monitors the solution for B and if
                                                                            % it increases (instead of decrease), returns it
                                                                            % back to the old B
        ConstVars.Monitor_Csol           = Config.monitor_csol;             % this options monitors the solution for C and if
                                                                            % it increases (instead of decrease), returns it
                                                                            % back to the old C
        % save settings and data
        if exist(Args.datafile, 'file')
            save(Args.datafile, '-append', 'ConstVars', 'Data');
        else
            save(Args.datafile, 'ConstVars', 'Data');
        end
        % continue interrupted learning process
        if Args.continue
            load(Args.datafile, 'B', 'C', 'w', 'iter');
            Data.W0    = B;
            Data.H0    = C;
            Data.w0    = w;
            Data.iter0 = iter;
        end
    % read data and initialize settings for feature extraction
    elseif strcmp(Args.what, 'extract')
        % get training parameters
        load(Args.datafile, 'ConstVars');
        if isempty(ConstVars)
            error(['Failed to load settings from datafile ''' Args.datafile ''' !!!']);
        end
        % read data
        [Data.V, Data.y, Dim] = gondola.ReadDataFromCOMPAREConfigFile(Args.imagelistfile, ConstVars.DownSampleRatio);
        % normalize data
        [Data.V, nrm_params] = gondola.normalizeData(Data.V, ConstVars);
    % otherwise, just load the previously saved settings and normalized data
    else
        load(Args.datafile, 'ConstVars', 'Data');
        if isempty(ConstVars)
            error(['Failed to load settings from datafile ''' Args.datafile ''' !!!']);
        end
        if isempty(Data)
            error(['Failed to load data from datafile ''' Args.datafile ''' !!!']);
        end
    end

    % all data is read from/written to this MAT file
    % this assignment MUST be after the load(Args.datafile) to overwrite the value
    % stored in the input .mat file !!
    ConstVars.DataFile = Args.datafile;
end

% --------------------------------------------------------------------------------------
% print the usage
function print_help()
    if isdeployed()
        usage   = 'gondola <learn|show|extract> [options]';
        program = 'program';
        prefix  = '  --';
        indent  = '    ';
    else
        usage   = 'gondola(''learn''|''show''|''extract''[, options...])';
        program = 'function';
        prefix  = '  ';
        indent  = '  ';
    end
    fprintf('Usage:\n');
    fprintf('  %s', usage);
    fprintf('\n');
    fprintf('Description:\n');
    fprintf('  This %s implements Generative-Discriminative Basis Learning\n', program);
    fprintf('  for Medical Imaging [1,2], a method referred to as GONDOLA. The subfunctions\n');
    fprintf('  implemented by gondola perform the learning of the basis vectors, the extraction\n');
    fprintf('  of features given a set of learned basis vectors, and a conversion of the basis\n');
    fprintf('  vectors to image files for visualization.\n');
    fprintf('\n');
    fprintf('  The first argument to this %s specifies which subfunction to perform:', program);
    fprintf('\n');
    fprintf('    learn     Learn basis vectors.\n');
    fprintf('    show      Save learned basis vectors as images.\n');
    fprintf('    extract   Extract features using learned basis vectors.\n');
    fprintf('\n');
    fprintf('  Following this argument are the different options as summarized below.\n');
    fprintf('\n');
    fprintf('Required arguments:\n');
    fprintf('%simagelistfile <file>        Input list file naming subject images and corresponding class labels.\n', prefix);
    fprintf('%s                            A text file containing list of subject images to be processed including a\n', indent);
    fprintf('%s                            some header information such as the number of images, image dimension, and\n', indent);
    fprintf('%s                            root data directory. Please refer to the software manual for details on the file\n', indent);
    fprintf('%s                            format (COMPARE format). The image list file must further contain class labels given \n', indent);
    fprintf('%s                            on the same line after the image file name. Notice that the class labels must be\n', indent);
    fprintf('%s                            positive integers, while class label 0 can be assigned to any unlabeled subject\n', indent);
    fprintf('%s                            in case of semi-supervised learning.\n', indent);
    fprintf('%sidlistfile <file>           Input list file naming IDs corresponding to subject images listed in image list.\n', prefix);
    fprintf('%s                            For each line in the imagelist file, except the first three header lines, this file\n', indent);
    fprintf('%s                            has to specify an ID for the corresponding input image. An ID can be a number or string.\n', indent);
    fprintf('%s                            This file is only required for the feature extraction.\n', indent);
    fprintf('\n');
    fprintf('Optional arguments:\n');
    fprintf('%sconfigfile <file>           Configuration file for the learning of the basis vectors.\n', prefix);
    fprintf('%s                            See section Configuration below for details.\n', indent);
    fprintf('%s                            (default: gondola.cfg)\n', indent);
    fprintf('%sdatafile <file>             Name of the data file (.mat) to write the learned basis vectors and\n', prefix);
    fprintf('%s                            the rest of optimization problem variables to if what is "learn".\n', indent);
    fprintf('%s                            Otherwise, this file specifies the .mat file from which to load these.\n', indent);
    fprintf('%s                            (default: gondola.mat)\n', indent);
    fprintf('%sfeaturesfile <file>         Name of output file in Weka format (.arff) for extracted features.\n', prefix);
    fprintf('%s                            The number of features is equal to the number of basis vectors plus 2,\n', indent);
    fprintf('%s                            where the last two spots are reserved for class labels and subject IDs.\n', indent);
    fprintf('%s                            (default: features.arff)\n', indent);
    fprintf('%soutputdir                   Output directory for any output files. In case of the resulting data file\n', prefix);
    fprintf('%s                            of the learned basis vectors or the file of the extracted features,\n', indent);
    fprintf('%s                            this directory is used to make given relative paths absolute. If absolute\n', indent);
    fprintf('%s                            file paths are given, this option has no effect. It is, however, in particular\n', indent);
    fprintf('%s                            of interest to specify an output directory for the basis images.\n', indent);
    fprintf('%s                            (default: cwd)\n', indent);
    fprintf('%scontinue                    Whether to continue processing from previous point during basis vector learning.\n', prefix);
    fprintf('%s                            This option is useful if the optimization process previously stopped before completion.\n', indent);
    fprintf('%s                            (default: false)\n', indent);
    fprintf('%srandseed <float>            Seed used for initialization of random number generator. (default: 0)\n', prefix);
    fprintf('%shelp                        Show help and exit.\n', prefix);
    fprintf('\n');
    fprintf('Configuration:\n');
    fprintf('  This %s requires a configuration file which specifies the parameters for the\n', program);
    fprintf('  particular experiment to perform. This file should be placed in the same directory as the\n');
    fprintf('  containing the results, respectively, the directory to which the results should be written to.\n');
    fprintf('  The format of the configuration file is as follows:\n');
    fprintf('\n');
    fprintf('    # this is a comment line\n');
    fprintf('    option: everything to the right of the colon is the value assigned to the option\n');
    fprintf('\n');
    fprintf('  Note that both the option name and value are case-insensitive.\n');
    fprintf('\n');
    fprintf('  The available configuration options are:\n');
    fprintf('    algo                     Specifies the algorithm. This can be one of the following:\n');
    fprintf('                             - MultiViewY\n');
    fprintf('                             - MultiViewXY\n');
    fprintf('                             - MultiViewXY_freeC\n');
    fprintf('                             where in case of MultiViewY there is one B per modality while in case of MultiViewXY\n');
    fprintf('                             one B is shared among modalities. In case of freeC, C is free to be any value, i.e.,\n');
    fprintf('                             including negative values.\n');
    fprintf('    csolver                  Solver to use for optmization of C (coefficients). Can be either SPG (Spectral Projected\n');
    fprintf('                             Gradient) or MOSEK. The default is to use SPG, which is also faster. MOSEK was initially\n');
    fprintf('                             used by GONDOLA and may still be preferred while extending GONDOLAs functionality due to\n');
    fprintf('                             its easier use and generality. For most users, the use of SPG is recommended, however.\n');
    fprintf('    downsampleratio          Specify the down-sampling ratio: use 1 or 2.\n');
    fprintf('                             Down-sampling speeds up execution by reducing the data dimension.\n');
    fprintf('    maxitr                   Maximum number of iterations per block; i.e. if MAXITR=30, the combined number\n');
    fprintf('                             of iterations for w, B, and C is set to 30 - i.e. 10 iterations per round.\n');
    fprintf('    numbasisvectors          Number of basis vectors (columns of B).\n');
    fprintf('    nrm_mode                 Specify normalization method.\n');
    fprintf('                             - 0   Do nothing.\n');
    fprintf('                             - 1   For each modality, z-score normalization; i.e. subtract the mean and divide result \n');
    fprintf('                                   by the standard deviation.\n');
    fprintf('                             - 2   For each modality, normalization between [0,1].\n');
    fprintf('                             - 3   For each modality, multiplication by a constant defined by "nrm_scale".\n');
    fprintf('                                   This is the recommended normalization method.\n');
    fprintf('    nrm_scale                If nrm_mode=3, this value is used for multiplication in normalization.\n');
    fprintf('    projectionmode           Projection mode to use for feature extraction. (default: ''inner_product'').\n');
    fprintf('    tol                      Tolerance value for optimization of B. It is recommended to use the default. (default: 0.5)\n');
    fprintf('    lambda_gen               Lambda coefficient to be multiplied with the generative term (lambda_1 in [1]).\n');
    fprintf('    lambda_disc              Lambda coefficient to be multiplied with the discriminative term (lambda_2 in [1]).\n');
    fprintf('    lambda_const             Lambda constraint to be used to specify sparsity in the feasible set (lambda_3  in [1]).\n');
    fprintf('    lambda_stab              Lambda of the stability. It is *NOT* recommended to change this. (default: 0.01).\n');
    fprintf('    saveaftereachiteration   If true, the program saves intermediate results every 3 hours. (default: false)\n');
    fprintf('\n');
    fprintf('  The following configuration options are advanced and it is recommended to use the default values.\n');
    fprintf('\n');
    fprintf('    bbmethod       Specifies how to approximate the hessian using Barzilai-Borwein method.\n');
    fprintf('                   Can take value 1, 2, or 3, but it is *NOT* recommended to change it from the\n');
    fprintf('                   default value of 3.\n');
    fprintf('    monitor_bsol   If true, the algorithm checks descend of the algorithm after optimizing B.\n');
    fprintf('                   If the cost function is not descending, B is not updated to the new value.\n');
    fprintf('    monitor_csol   If true, the algorithm checks descend of the algorithm after optimizing C.\n');
    fprintf('                   If the cost function is not descending; C is not updated to the new value.\n');
    fprintf('\n');
    fprintf('  For example, a typical configuration file looks as follows:\n');
    fprintf('\n');
    fprintf('  # this is a comment line \n');
    fprintf('  algo:                   MultiViewXY\n');
    fprintf('  bbmethod:               3\n');
    fprintf('  csolver:                SPG\n');
    fprintf('  downsampleratio:        2\n');
    fprintf('  lambda_gen:             10\n');
    fprintf('  lambda_disc:            0.1\n');
    fprintf('  lambda_const:           0.2\n');
    fprintf('  lambda_stab:            0.01\n');
    fprintf('  maxitr:                 300\n');
    fprintf('  monitor_bsol:           false\n');
    fprintf('  monitor_csol:           false\n');
    fprintf('  nrm_mode:               3\n');
    fprintf('  nrm_scale:              5\n');
    fprintf('  numbasisvectors:        30\n');
    fprintf('  saveaftereachiteration: true\n');
    fprintf('  tol:                    0.5\n');
    fprintf('\n');
    fprintf('Examples:\n');
    if isdeployed()
    fprintf('  gondola learn --configfile /path/to/gondola.cfg --imagelistfile /path/to/images.lst --idlistfile /path/to/ids.lst\n');
    else
    fprintf('  gondola(''learn'', ''configfile'', ''/path/to/gondola.cfg'', ''imagelistfile'', ''/path/to/images.lst'', ''idlistfile'', ''/path/to/ids.lst'');\n');
    end
    fprintf('\n');
    fprintf('      Learns the basis vectors for the subject images named in images.lst with corresponding IDs given\n');
    fprintf('      by the ids.lst file. The configuration for the process is read from the file gondola.cfg\n');
    fprintf('      Both, intermediate and final, results are saved to the gondola.mat MATLAB data file.\n');
    fprintf('      If you want to specify a different output name for the results file, use the datafile option.\n');
    fprintf('      Since the continue option is omitted, the process will begin from scratch.\n');
    fprintf('\n');
    if isdeployed()
    fprintf('  gondola show --datafile /path/to/gondola.mat --outputdir /path/to/basis/images\n');
    else
    fprintf('  gondola(''show'', ''datafile'', ''/path/to/gondola.mat'', ''outputdir'', ''/path/to/basis/images'');\n');
    end
    fprintf('\n');
    fprintf('      Reads the learned basis vectors from the file /path/to/gondola.mat and saves them as\n');
    fprintf('      images to the directory /path/to/basis/images.\n');
    fprintf('\n');
    if isdeployed()
    fprintf('  gondola extract --datafile /path/to/gondola.mat --imagelistfile /path/to/images.lst --idlistfile /path/to/ids.lst\n');
    else
    fprintf('  gondola(''extract'', ''datafile'', ''/path/to/gondola.mat'', ''imagelistfile'', ''/path/to/images.lst'', ''idlistfile'', ''/path/to/ids.lst'');\n');
    end
    fprintf('\n');
    fprintf('      Reads the results of the learning stage saved in /path/to/gondola.mat, extracts the features\n');
    fprintf('      and saves them in the file gondola.arff which can be further processed/visualized using Weka.\n');
    fprintf('      If you want to specify a different output name for the features file, use the featuresfile option.\n');
    fprintf('\n');
    fprintf('Refrences:\n');
    fprintf('  [1] K. N. Batmanghelich, B. Taskar, C. Davatzikos; Generative-Discriminative\n');
    fprintf('      Basis Learning for Medical Imaging; IEEE Trans Med Imaging. 2012 Jan;31(1):51-69. Epub 2011 Jul 25\n');
    fprintf('\n');
    fprintf('  [2] K. N. Batmanghelich, B. Taskar, D. Ye, C. Davatzikos; Regularized Tensor\n');
    fprintf('      factorization for multi-modality medical image classification, MICCAI 2011, LNCS 6893, p17\n');
    fprintf('\n');
    fprintf('Contact:\n');
    fprintf('  SBIA Group <sbia-software at uphs.upenn.edu>\n');
end

% Copyright (c) 2011, 2012 University of Pennsylvania. All rights reserved.
% See http://www.rad.upenn.edu/sbia/software/license.html or COPYING file.
%
% Contact: SBIA Group <sbia-software at uphs.upenn.edu>
