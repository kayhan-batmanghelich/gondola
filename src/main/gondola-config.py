#! /usr/bin/env python

##############################################################################
# @file  gondola-config.py
# @brief Configure cross-validation experiments.
#
# Copyright (c) 2012 University of Pennsylvania. All rights reserved.
# See http://www.rad.upenn.edu/sbia/software/license.html or COPYING file.
#
# Contact: SBIA Group <sbia-software at uphs.upenn.edu>
##############################################################################

import os
import errno
import sys
import textwrap

from gondola import basis


# ============================================================================
# constants
# ============================================================================

## @brief String used as prefix for each first line of an error message.
ERRMSG_PREFIX = "** error: "

# ============================================================================
# help
# ============================================================================

# ----------------------------------------------------------------------------
## @brief Print help of specified configuration option.
def print_config_help(o):
    if o in CONFIG_HELP:
        sys.stdout.write("  --%s" % o.lower())
        sys.stdout.write(textwrap.fill(textwrap.dedent(CONFIG_HELP[o]), 90,
                                       initial_indent    = ' ' * (29 - len(o) - 4),
                                       subsequent_indent = ' ' * 30))
        sys.stdout.write("\n")
    else: raise Exception("Unknown configuration option: %s\n" % o)

# ----------------------------------------------------------------------------
## @brief Print usage information including help of available configuration options.
def print_help():
    print "Usage:\n  %(EXENAME)s [options] <images> <ids>" % {'EXENAME': basis.exename() }
    print """
Required arguments:
  <images>                    Name of text file listing the image files including a three
                              line header stating the number of images, the common image
                              dimension, and the root path for the image files which is used
                              to make relative image file paths absolute. If the root path
                              itself is relative, it is interpreted relative to the directory
                              containing the <images> list file. Note that the format of this
                              input file corresponds to the one used by the COMPARE software
                              which is developed and distributed by SBIA.
  <ids>                       Name of text file listing the IDs corresponding to the images
                              named in the images list given by the <images> file."""
    for o, a in sorted(DEFAULT_CONFIG.items()):
        if a is None: print_config_help(o)
    print """
Optional arguments:
  -d  --expdir <dir>          Directory for experiment configuration and output files.
                              For each fold, a subdirectory is created within this directory.
                              (default: cwd)
  -n  --numfolds <int>        Number of folds, i.e., cross-validation rounds. (default: 10)
  -s  --shuffle               Shuffle the input lists. (default: off)
  -r  --regress               Divide entire list into folds rather than individual classes.
                              Use this option to create folds for regression. (default: off)

Configuration:
  Single optional configuration values can be changed on the command-line using the
  following options, where the option name corresponds to the configuration name
  (case-insensitive) prefixed by two dashes (--).
"""
    for o, a in sorted(DEFAULT_CONFIG.items()):
        if not a is None: print_config_help(o)
    print """
Example:
  %(EXENAME)s --expdir ADHD_Competition/Structural/exp1708_5foldCV
           --algo MultiViewXY_BoxedSparsity --numfolds 5 --numbasisvectors 30
           --lambda_gen 10 --lambda_disc 0.1 --lambda_const 0.2 images.lst ids.lst

    Creates a directory structure for the cross-validation experiments including configuration
    files for the gondola and gondola-crossval executables in the
    ADHD_Competition/Structural/exp1708_5foldCV directory. For the cross-validation, 5 folds
    are generated from the set of input images named by the images.lst with corresponding
    subject IDs in ids.lst. An example of these files can be found in the example/ directory of
    the GONDOLA package. Moreover, non-default configuration values for the lambda options
    as well as the number of basis vectors are specified above using the corresponding options.
    Note that the specification of the number of basis vectors is mandatory.

Refrences:
  [1] K. N. Batmanghelich, B. Taskar, C. Davatzikos; Generative-Discriminative
      Basis Learning for Medical Imaging; IEEE Trans Med Imaging. 2012 Jan;31(1):51-69. Epub 2011 Jul 25

  [2] K. N. Batmanghelich, B. Taskar, D. Ye, C. Davatzikos; Regularized Tensor
      factorization for multi-modality medical image classification, MICCAI 2011, LNCS 6893, p17
""" % {'EXENAME': basis.exename() }
    basis.print_contact()

# ============================================================================
# gondola configuration
# ============================================================================

## @brief Array of valid choices/values for @c algo configuration option.
ALGOS = ('MultiViewY',
         'MultiViewXY',
         'MultiViewXY_freeC')

## @brief Array of valid choices/values for @c csolver configuration option.
CSOLVERS = ('SPG', 'MOSEK')

## @brief Help of configuration option.
#
# This dictionary contains a short help for each configuration option which is
# used to generate the full help output of this program. Note that any newline
# characters in the help string are ignored as the help will be reformatted as
# needed using the textwrap.fill() function.
#
# @note This dictionary is also used in order to test if a given option name
#       is a valid configuration, i.e., there @b must be an entry for each
#       configuration option that can be set on the command-line.
CONFIG_HELP = {
    # attention: make sure that all lines of triple-quoted strings are lined up,
    #            i.e., always start with the help text in a new line after
    #            the opening """ triple-quotes. Otherwise, the textwrap.dedent()
    #            function does not remove the leading indentation properly.
    'algo':                   """
                              Type of algorithm that should be applied: %(algos)s.
                              In case of MultiViewY, there is one set of basis vectors
                              per modality while in case of MultiViewXY a common set of
                              the basis vectors is shared among modalities.
                              In case of freeC, C is free to be any value, including
                              negative values."""
                                % {'algos': ', '.join(ALGOS)},
    'csolver':                """
                              Solver to use for optmization of C (coefficients). Can be either
                              SPG (Spectral Projected Gradient) or MOSEK. The default is
                              to use SPG, which is also faster. MOSEK was initially used by
                              GONDOLA and may still be preferred while extending GONDOLAs
                              functionality due to its easier use and generality. For most
                              users, the use of SPG is recommended, however.
                              """,
    'downsampleratio':        """
                              Specify the down-sampling ratio: use 1 or 2. Down-sampling speeds
                              up execution by reducing the data dimension.
                              """,
    'nrm_mode':               """
                              Specify normalization method: 0: do nothing,
                              1: for each modality, z-score normalization; i.e. subtract the
                              mean and divide result by the standard deviation, 2: for each
                              modality, normalization between [0,1], 3: for each modality,
                              multiplication by a constant defined by "nrm_scale" This is the
                              recommended normalization method.
                              """,
    'nrm_scale':              """
                              If nrm_mode=3, this value is used for multiplication in normalization.
                              """,
    'maxitr':                 """
                              Maximum number of iterations per block; i.e. if 30, the combined number
                              of iterations for w, B, and C is set to 30, i.e., 10 iterations per round.
                              """,
    'tol':                    """
                              Tolerance value for optimization of B. It is recommended to use the default.
                              """,
    'numbasisvectors':        """
                              Number of basis vectors (columns of B).
                              """,
    'lambda_gen':             """
                              Lambda coefficient to be multiplied with the generative term (lambda_1 in [1]).
                              """,
    'lambda_disc':            """
                              Lambda coefficient to be multiplied with the discriminative term (lambda_2 in [1]).
                              """,
    'lambda_const':           """
                              Lambda constraint to be used to specify sparsity in the feasible
                              set (lambda_3  in [1]).
                              """,
    'lambda_stab':            """
                              Lambda of the stability. It is *NOT* recommended to change this. (default: 0.01)
                              """,
    'bbmethod':               """
                              Specifies how to approximate the hessian using Barzilai-Borwein method. (default: 3)
                              """,
    'monitor_bsol':           """
                              If true, the algorithm checks descend of the algorithm after optimizing B.
                              If the cost function is not descending, B is not updated to the new value.
                              """,
    'monitor_csol':           """
                              If true, the algorithm checks descend of the algorithm after optimizing C.
                              If the cost function is not descending; C is not updated to the new value.
                              """,
    'saveaftereachiteration': """
                              If true, the program saves intermediate results every 3 hours. (default: false)
                              """,
}

## @brief Default configuration values.
DEFAULT_CONFIG = {
    'algo':                   None,
    'bbmethod':               3,
    'csolver':                'SPG',
    'downsampleratio':        2,
    'lambda_gen':             10,
    'lambda_disc':            0.1,
    'lambda_const':           0.2,
    'lambda_stab':            0.01,
    'maxitr':                 300,
    'monitor_bsol':           False,
    'monitor_csol':           False,
    'numbasisvectors':        None,
    'nrm_mode':               3,
    'nrm_scale':              5,
    'saveaftereachiteration': True,
    'tol':                    0.5,
}

# ----------------------------------------------------------------------------
## @brief Write configuration file for gondola.
def write_gondola_config(filename, config):
    f = open(filename, 'wt')
    if not f: raise Exception("%sFailed to create file %s!" % (ERRMSG_PREFIX, filename))
    maxlen = 0
    for o in config.keys():
        if len(o) > maxlen: maxlen = len(o)
    for o, a in sorted(config.items()):
        f.write("%s: %s%s\n" % (o, ' ' * (maxlen - len(o)), a))
    f.close()

# ============================================================================
# cross-validation experiment
# ============================================================================

## @brief Default cross-validation experiment configuration.
CROSSVAL_CONFIG = """\
[commands]
learn:            %(gondola)s learn
                      --configfile        "%%(configfile)s"
                      --imagelistfile     "%%(imagelistfile)s"
                      --datafile          "%%(datafile)s"
show:             %(gondola)s show
                      --imagelistfile     "%%(imagelistfile)s"
                      --datafile          "%%(datafile)s"
                      --outputdir         "%%(basisimagedir)s"
extract:          %(gondola)s extract
                      --imagelistfile     "%%(imagelistfile)s"
                      --idlistfile        "%%(idlistfile)s"
                      --datafile          "%%(datafile)s"
                      --featuresfile      "%%(featuresfile)s"
search:           wekaParamSearchForClassifier -i -w
                      --arffFile          "%%(featuresfile)s"
                      --csvFile           "%%(bestparamsfile)s"
                      --listOfClassifiers "%%(classifiers)s"
classify:         wekaClassifier
                      --trainArff         "%%(training.featuresfile)s"
                      --testArff          "%%(testing.featuresfile)s"
                      --bestClassifier    "%%(classifier)s"
                      --bestParam         "%%(bestparams)s"
                      --extraParam        "%%(extraparams)s"
                      --trainCSV          "%%(training.resultfile)s"
                      --testCSV           "%%(testing.resultfile)s"
                      --hdrTrain          "%%(training.resultheader)s"
                      --hdrTest           "%%(testing.resultheader)s"
jobstat:          qstat -j %%(jobid)s

[settings]
foldids:          %(foldids)s
classifiers:      Logistic, SMO, Simple Logistic, Bayesian, Random Forest
summaryfile:      SummaryFor%%(classifier)sClassifier.txt

[training]
configfile:       gondola.cfg
imagelistfile:    %%(foldid)s/training.lst
idlistfile:       %%(foldid)s/trainids.lst
datafile:         %%(foldid)s/gondola.mat
basisimagedir:    %%(foldid)s/
featuresfile:     %%(foldid)s/training.arff
bestparamsfile:   %%(foldid)s/bestparams.csv
resultfile:       %%(foldid)s/%%(classifier)sClassifierTrainResults.csv
resultcolumn:     ClassLabel

[testing]
imagelistfile:    %%(foldid)s/testing.lst
idlistfile:       %%(foldid)s/testids.lst
featuresfile:     %%(foldid)s/testing.arff
resultfile:       %%(foldid)s/%%(classifier)sClassifierTestResults.csv
resultcolumn:     ClassLabel
"""

# ----------------------------------------------------------------------------
## @brief Write configuration file for gondola-crossval.
def write_crossval_config(filename, numfolds, gondola='gondola'):
    f = open(filename, 'wt')
    if not f: raise Exception("%sFailed to create file %s!" % (ERRMSG_PREFIX, filename))
    f.write(CROSSVAL_CONFIG % {'foldids': ', '.join([str(i) for i in range(1, numfolds + 1)]),
                               'gondola': gondola})
    f.close()

# ============================================================================
# main
# ============================================================================

# ----------------------------------------------------------------------------
## @brief Case-insensitive check if string is in given set of choices.
#
# @returns The choise with preferred case if valid or @c None otherwise.
def choice(string, choices):
    for choice in choices:
        if string.lower() == choice.lower(): return choice
    return None

# ----------------------------------------------------------------------------
## @brief Creates the directory structure and .config files.
def main(images_list_file, ids_list_file,
         expdir=None, numfolds=10, config=DEFAULT_CONFIG,
         regress=False, shuffle=False, gondola='gondola'):
    errmsg = '' # collect all error messages in one string and throw a single
                # exception containing all errors if any
    # create "union" of user input and default configuration
    cfg = DEFAULT_CONFIG
    cfg.update(config)
    # check if all configuration values are valid and convert values to strings
    if 'algo' in config and choice(config['algo'], ALGOS) is None:
        errmsg = '\n\n'.join([errmsg, "%sInvalid value for 'algo' configuration: %s\n\n    The available choices are:\n\n    - %s" % (ERRMSG_PREFIX, config['algo'], '\n    - '.join(ALGOS))])
    if 'csolver' in config and choice(config['csolver'], CSOLVERS) is None:
        errmsg = '\n\n'.join([errmsg, "%sInvalid value for 'csolver' configuration: %s\n\n    The available choices are:\n\n    - %s" % (ERRMSG_PREFIX, config['csolver'], '\n    - '.join(CSOLVERS))])
    for o in sorted(cfg.keys()):
        if cfg[o] is None: errmsg = '\n\n'.join([errmsg, "%sMissing '%s' configuration!" % (ERRMSG_PREFIX, o)])
        else:              cfg[o] = str(cfg[o])
    # raise Exception with all collected errors as message
    if errmsg: raise Exception('\n\n\n'.join([errmsg.lstrip('\n'), "    See --help for usage information and a list of required and available options."]))
    # make experiment directory and change to it
    if not expdir: expdir = os.getcwd()
    if os.path.isfile(expdir):
        raise Exception("%sA file named '%s' already exists! Cannot create experiment directory of same name.\n\n    Please specify a different directory for the experiment." % (ERRMSG_PREFIX, expdir))
    if os.path.isdir(expdir):
        if len(os.listdir(expdir)) > 0:
            raise Exception("%sThe directory '%s' already exists and is not empty!\n\n    Please specify a different directory for the experiment or remove this directory first." % (ERRMSG_PREFIX, expdir))
    else:
        try:
            os.makedirs(expdir)
        except os.error, e:
            raise Exception("%sFailed to create directory for experiment:\n\n    %s" % (ERRMSG_PREFIX, str(e)))
    try:
        os.chdir(expdir)
    except os.error, e:
        raise Exception("%sFailed to change into experiment directory:\n\n    %s", (ERRMSG_PREFIX, str(e)))
    # create list files for folds
    cmd = ['createFoldsFromIdFnLists', '-f', images_list_file, '-i', ids_list_file, '-n', numfolds]
    if shuffle: cmd.append('-s')
    if regress: cmd.append('-r')
    if basis.execute(cmd, allow_fail=True) != 0:
        raise Exception("%sFailed to setup directory structure and create training and testing lists!" % ERRMSG_PREFIX)
    # generate configuration file for gondola
    cfg['algo']    = choice(cfg['algo'],    ALGOS)
    cfg['csolver'] = choice(cfg['csolver'], CSOLVERS)
    write_gondola_config(os.path.join(expdir, 'gondola.cfg'), cfg)
    # generate configuration file for gondola-crossval
    write_crossval_config(os.path.join(expdir, 'crossval.cfg'), numfolds, gondola)

# ----------------------------------------------------------------------------
# parse command-line arguments and pass them on to the main() function
if __name__ == '__main__':
    if len(sys.argv) == 1:
        print_help()
        sys.exit(1)

    expdir           = None
    ids_list_file    = None
    images_list_file = None
    config           = {}
    numfolds         = 10
    regress          = False
    shuffle          = False
    gondola          = 'gondola'

    i = 1
    while i < len(sys.argv):
        # current option and argument
        o = sys.argv[i]
        if i + 1 < len(sys.argv): a = sys.argv[i + 1]
        else:                     a = None
        # process option/argument
        if o in ('-h', '--help', '--helpshort'):
            print_help()
            sys.exit(0)
        elif o in ('-v', '--verbose'):
            pass
        elif o == '--version':
            basis.print_version('gondola-config')
            sys.exit(0)
        elif o in ('-d', '--expdir'):
            if not a:
                sys.stderr.write("Option -d (--expdir) requires an argument!\n")
                sys.exit(1)
            expdir = a
            i += 1
        elif o in ('-n', '--numfolds'):
            if not a:
                sys.stderr.write("Option -n (--numfolds) requires an argument!\n")
                sys.exit(1)
            numfolds = int(a)
            i += 1
        elif o in ('-s', '--shuffle'):
            shuffle = True
        elif o in ('-r', '--regress'):
            regress = True
        elif o == '--gondola':
            gondola = a
            i += 1
        elif o == '--':
            i += 1
            break
        elif o.startswith('--'):
            c = o[2:].lower()
            for k in CONFIG_HELP.keys():
                if k.lower() == c:
                    c = k
                    break
            if not c in CONFIG_HELP:
                sys.stderr.write("Invalid option: %s! See --help for a list of available options.\n" % o)
                sys.exit(1)
            if not a:
                sys.stderr.write("Option %s requires an argument!\n" % o)
                sys.exit(1)
            config[c] = a
            i += 1
        elif not images_list_file:
            images_list_file = o
        elif not ids_list_file:
            ids_list_file = o
        else:
            sys.stderr.write("Invalid argument: %s. See --help for usage information.\n" % o)
            sys.exit(1)
        # next argument
        i += 1

    if i == len(sys.argv) - 1 and not ids_list_file:
        ids_list_file    = sys.argv[i]
    elif i == len(sys.argv) - 2 and not images_list_file and not ids_list_file:
        images_list_file = sys.argv[i    ]
        ids_list_file    = sys.argv[i + 1]
    elif i < len(sys.argv):
        sys.stderr.write("Too many arguments! See --help for usage information.\n")
        sys.exit(1)

    if not images_list_file:
        sys.stderr.write("No images list file specified!\n")
        sys.exit(1)
    if not ids_list_file:
        sys.stderr.write("No IDs list file specified!\n")
        sys.exit(1)

    try:
        main(images_list_file, ids_list_file, expdir, numfolds, config, regress, shuffle, gondola)
    except Exception, e:
        sys.stderr.write("\n%s\n\n" % str(e))
        sys.exit(1)
    sys.exit(0)
