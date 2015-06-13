% This function solves the following optimization problem with MOSEK
%                       \min_{C}     \lambda_gen*\| V - B*C \|_{F}^{2} + \lambda_stab* \| C \|_{F}^{2}
%         It assumes that V is D x N matrix and B is a D x R matrix and hence
%         C becomes  R x N matrix
function [C,Report]  = CSolver_freeC_mosek(C0,V,B,w,y,options)
    % initialization and some required variables
    N = size(C0,2) ;
    r = size(C0,1) ; 
    lambda_gen = options.lambda_gen ;
    lambda_stab = options.lambda_stab ;           
    % main body of function 
    % quadratic term
    [rows,cols,vals] = gondola.blockdiag(double(B'*B),N,0) ;
    rows = rows + 1 ;    % +1 must be added because C index from zero and MATLAB index from 1
    cols = cols + 1 ;    % +1 must be added because C index from zero and MATLAB index from 1
    A = sparse(rows,cols,vals) ;
    Q = lambda_gen*2*A ;            % it has to be multiplied by 2 because MOSEK has extra 0.5 in its objective
    Q = Q + lambda_stab*2*speye(r*N) ;
    % linear term
    K = V'*B ;
    [rows,cols,vals] = gondola.blockdiag(double(K),N,0) ;
    rows = rows + 1 ;    % +1 must be added because C index from zero and MATLAB index from 1
    cols = cols + 1 ;    % +1 must be added because C index from zero and MATLAB index from 1
    A = sparse(rows,cols,vals) ;
    tmpeye = speye(N) ;
    e = -2*tmpeye(:)'*A ;                % linear term. It is equal to  -2*trace(V' B C).
    e = lambda_gen*e' ;
    % constraints  (no constraint in this case)
    blx   = [];
    bux   = [];
    % linear equality constraints
    a = speye(r*N) ;
    blc = [] ;
    buc = [] ;
    % Optimize the problem.
    %[res] = mskqpopt(Q,e,a,blc,buc,blx,bux);
    clear prob
    prob.c = e ;
    prob.a = a ;
    prob.bux = bux ;
    prob.blx = blx ;
    prob.buc = buc ;
    prob.blc = blc ;
    [prob.qosubi, prob.qosubj, prob.qoval] = find(tril(Q)) ;
    [rcode,res] = mosekopt('minimize',prob);
    C = reshape(res.sol.itr.xx,r,N) ;
    Report = rcode ;
end

% Copyright (c) 2011, 2012 University of Pennsylvania. All rights reserved.
% See http://www.rad.upenn.edu/sbia/software/license.html or COPYING file.
%
% Contact: SBIA Group <sbia-software at uphs.upenn.edu>
