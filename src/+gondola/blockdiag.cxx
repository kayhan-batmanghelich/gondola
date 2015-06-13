/**
 * @file  blockdiag.cxx
 * @brief Build block-diagonal matrix.
 *
 * Copyright (c) 2012 University of Pennsylvania. All rights reserved.
 * See http://www.rad.upenn.edu/sbia/software/license.html or COPYING file.
 *
 * Contact: SBIA Group <sbia-software at uphs.upenn.edu>
 */

#include "mex.h"
#include "math.h"
#include "matrix.h"


#include <iostream>
#include <vector>
using namespace std;


class CblockDiag
{
    double                              *m_inMatrix ;
    int                                 m_numRep ;
    int                                 m_inMatrix_nrows ;
    int                                 m_inMatrix_ncols ;
    bool                                m_mode ;
    vector<unsigned long int>           *m_rows ;
    vector<unsigned long int>           *m_cols ;
    vector<double>                      *m_vals ;
public:
    // constructor
    CblockDiag(double *inMatrix, int inMatrix_nrow, int inMatrix_ncols, int numRep, bool m)
    {
        m_mode = m ;
        m_inMatrix = inMatrix ;
        m_inMatrix_nrows = inMatrix_nrow ;
        m_inMatrix_ncols = inMatrix_ncols ;
        m_numRep = numRep ;
        m_rows = new vector<unsigned long int> ;
        m_cols = new vector<unsigned long int> ;
        m_vals = new vector<double>  ;
    }
    
    // de-constructor
    ~CblockDiag()
    {
        // deconstruction
        delete m_rows ;
        delete m_cols ;
        delete m_vals ;
    }
    
    // build matrix
    void buildMatrix()
    {
        if (!(m_mode))
        {
            for (double r=0;r<m_numRep; r++)
            {
                for (double j=0;j<m_inMatrix_ncols;j++)
                {
                    for (double i=0;i<m_inMatrix_nrows;i++)
                    {
                        unsigned long int  ind = (unsigned long int)(((double)m_inMatrix_nrows)*j  + i) ;
                        unsigned long int  row = (unsigned long int)( i + r*((double) m_inMatrix_nrows)) ; 
                        unsigned long int  col = (unsigned long int)( j + r*((double) m_inMatrix_ncols)) ; 
                        m_vals->push_back(m_inMatrix[ind]) ;
                        m_rows->push_back(row) ;
                        m_cols->push_back(col) ;
                    }
                }
            }
        }
        else
        {
            for (double r=0;r<m_numRep; r++)
            {
                for (double j=0;j<m_inMatrix_ncols;j++)
                {
                    for (double i=0;i<m_inMatrix_nrows;i++)
                    {
                        unsigned long int  ind = (unsigned long int)(((double)m_inMatrix_nrows)*j  + i) ;
                        unsigned long int  row = (unsigned long int)( r + ((double)m_numRep*i)) ; 
                        unsigned long int  col = (unsigned long int)( r + ((double)m_numRep*j)) ; 
                        m_vals->push_back(m_inMatrix[ind]) ;
                        m_rows->push_back(row) ;
                        m_cols->push_back(col) ;
                    }
                }
            }
        }
    }
    
    // return number of non-zero value
    unsigned long int GetNnz()
    {
        unsigned long int tmp ;
        tmp = (unsigned long int)(m_vals->size()) ;
        return tmp ;
    }
    
    // copy columns
    void CopyCols(double *cols)
    {
        for (unsigned long int i=0;i<GetNnz();i++)
        {
            cols[i] = double((*m_cols)[i]) ;
        }
    }
    
    // copy rows
    void CopyRows(double *rows)
    {
        for (unsigned long int i=0;i<GetNnz();i++)
        {
            rows[i] = double((*m_rows)[i]) ;
        }
    }
    
    // copy weights
    void CopyValues(double *w)
    {
        for (unsigned long int i=0;i<GetNnz();i++)
        {
            w[i] = (*m_vals)[i] ;
        }
    }
    
   
} ;



// notation :
//	[rows,cols,vals] = blockDiag(K,R,mode)
/* the gateway function */
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
{
    // make sure there are enough inputs and outputs
    if(nrhs!=3) 
    	mexErrMsgTxt("three inputs are required.");
    if(nlhs!=3) 
    	mexErrMsgTxt("three outputs are required.");


    // repeat information
    int numRep = (int)mxGetScalar(prhs[1]) ;
    if (!(numRep > 0) )
    {
        mexErrMsgTxt("number of repeats should be positive ") ;
        
    }
    
    // mode information
    int m = (int)mxGetScalar(prhs[2]) ;
    bool mode ;
    if (!((m==0) || (m==1)))
    {
        mexErrMsgTxt("mode should be either zero or one ") ;
        
    }
    mode = (m==1) ;
    
    // input matrix information
    if (mxGetClassID(prhs[0]) != mxDOUBLE_CLASS )
    {
          mexErrMsgTxt("input matrix must be double, other types are not supperted yet !") ;
    }
    double *inMatrix = mxGetPr(prhs[0]) ;
    int  numcols    = (int)mxGetN(prhs[0]) ;
    int  numrows    = (int)mxGetM(prhs[0]) ;
    if (mode)
    {
        if (numcols!=numrows)
        {
            mexErrMsgTxt("for mode = True, number of columns and rows should be equal !! ") ;
        }
    }
    
    // initialize the class
    CblockDiag        blockDiag(inMatrix, numrows, numcols, numRep,mode) ;
    blockDiag.buildMatrix() ;
    cout << "Done with Matrix building ...." << endl ;  
 
   // make sure that size of matrices matches
    mwSize   M ;
    M = (mwSize)blockDiag.GetNnz();
    // alocate memory
    plhs[0] = mxCreateDoubleMatrix(M,1, mxREAL);
    double *outMatrix_rows = mxGetPr(plhs[0]) ;
    blockDiag.CopyRows(outMatrix_rows) ;
    
    plhs[1] = mxCreateDoubleMatrix(M,1, mxREAL);
    double *outMatrix_cols = mxGetPr(plhs[1]) ;
    blockDiag.CopyCols(outMatrix_cols) ;
    
    plhs[2] = mxCreateDoubleMatrix(M,1, mxREAL);
    double *outMatrix_vals = mxGetPr(plhs[2]) ;
    blockDiag.CopyValues(outMatrix_vals) ;

    
}
