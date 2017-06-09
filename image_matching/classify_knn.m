function [ E, A, S, d ] = classify_knn( P, Pnames, K, X, Xnames, varargin )
%
% CLASSIFY_KNN: Classify using K Nearest Neighbor
%
% Usage:
%   [ E, A, S, dists ] = classify_knn( P, Plabels, K, X, Xlabels, ... )
%
% Input:
%   P        - Prototypes data matrix. Each column vector is a data point.
%   Plabels  - Prototypes labels.
%   K        - K neighbors.
%   X        - Testing data matrix. Each column vector is a data point.
%   Xlabels  - Testing data class labels.
%
% Input (optional):
%   'perclass',(true|false)   - Compute error/score for each class (default=false)
%   'euclidean'               - Euclidean distance (default=true)
%   'cosine'                  - Cosine distance (default=false)
%   'hamming'                 - Hamming distance (default=false)
%   'tangent'                 - Tangent distance (default=false)
%   'rtangent'                - Ref. tangent distance (default=false)
%   'otangent'                - Obs. tangent distance (default=false)
%   'atangent'                - Avg. tangent distance (default=false)
%   'tangVp',tangVp           - Tangent bases of prototypes
%   'tangVx',tangVx           - Tangent bases of testing data
%
% Output:
%   E        - Classification error
%   A        - Assigned class
%   S        - Classification score
%   dists    - Pairwise distances
%
% $Revision: 197 $
% $Date: 2014-06-30 11:16:48 +0200 (Mon, 30 Jun 2014) $
%

% Copyright (C) 2008-2012 Mauricio Villegas <mauvilsa@upv.es>
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program. If not, see <http://www.gnu.org/licenses/>.

fn = 'classify_knn:';
minargs = 5;

if ischar(P)
  unix(['echo "$Revision: 197 $* $Date: 2014-06-30 11:16:48 +0200 (Mon, 30 Jun 2014) $*" | sed "s/^:/' fn ' revision/g; s/ : /[/g; s/ (.*)/]/g;"']);
  return;
end

A = [];
S = [];

[ D, Np ] = size(P);
K = min(Np,K);
Nx = size(X,2);

perclass = false;
torthonorm = false;

dtype.euclidean = true;
dtype.cosine = false;
dtype.tangent = false;
dtype.rtangent = false;
dtype.otangent = false;
dtype.atangent = false;
dtype.hamming = false;

logfile = 2;

n = 1;
argerr = false;
while size(varargin,2)>0
  if ~ischar(varargin{n})
    argerr = true;
  elseif strcmp(varargin{n},'perclass') || ...
         strcmp(varargin{n},'torthonorm')
    eval([varargin{n},'=varargin{n+1};']);
    if ~islogical(varargin{n+1})
      argerr = true;
    else
      n = n+2;
    end
  elseif strcmp(varargin{n},'euclidean') || ...
         strcmp(varargin{n},'tangent') || ...
         strcmp(varargin{n},'rtangent') || ...
         strcmp(varargin{n},'otangent') || ...
         strcmp(varargin{n},'atangent') || ...
         strcmp(varargin{n},'hamming') || ...
         strcmp(varargin{n},'cosine')
    dtype.euclidean = false;
    dtype.cosine = false;
    dtype.tangent = false;
    dtype.rtangent = false;
    dtype.otangent = false;
    dtype.atangent = false;
    dtype.hamming = false;
    eval(['dtype.',varargin{n},'=true;']);
    n = n+1;
  elseif strcmp(varargin{n},'tangVp') || ...
         strcmp(varargin{n},'tangVx')
    eval([varargin{n},'=varargin{n+1};']);
    if ~isnumeric(varargin{n+1})
      argerr = true;
    else
      n = n+2;
    end
  else
    argerr = true;
  end
  if argerr || n>size(varargin,2)
    break;
  end
end

if argerr
  fprintf(logfile,'%s error: incorrect input argument %d (%s,%g)\n',fn,n+minargs,varargin{n},varargin{n+1});
  return;
elseif nargin-size(varargin,2)~=minargs
  fprintf(logfile,'%s error: not enough input arguments\n',fn);
  return;
elseif size(X,1)~=D
  fprintf(logfile,'%s error: dimensionality of prototypes and data must be the same\n',fn);
  return;
elseif size(Plabels,2)~=Np || size(Plabels,1)~=1 || ...
      ( sum(size(Xlabels))~=0 && ( size(Xlabels,2)~=Nx || size(Xlabels,1)~=1 ) )
  fprintf(logfile,'%s error: labels must have the same size as the number of data points\n',fn);
  return;
elseif ~exist('tangVp','var') && ( dtype.tangent || dtype.atangent || dtype.rtangent )
  fprintf(logfile,'%s error: tangents of prototypes should be given\n',fn);
  return;
elseif ~exist('tangVx','var') && ( dtype.tangent || dtype.atangent || dtype.otangent )
  fprintf(logfile,'%s error: tangents of testing data should be given\n',fn);
  return;
elseif ( exist('tangVp','var') && mod(size(tangVp,2),Np)~=0 ) || ...
       ( exist('tangVx','var') && mod(size(tangVx,2),Nx)~=0 )
  fprintf(logfile,'%s error: number of tangents should be a multiple of the number of samples\n',fn);
  return;
end

if exist('tangVp','var') && ( dtype.rtangent || dtype.atangent || dtype.tangent )
  cfg.torthonorm = torthonorm;
  cfg.tangVp = tangVp;
end
if exist('tangVx','var') && ( dtype.otangent || dtype.atangent || dtype.tangent )
  cfg.torthonorm = torthonorm;
  cfg.tangVx = tangVx;
end

cfg.dtype = dtype;
d = classify_knn_dstmat(P,X,cfg);

Clabels = unique(Plabels);
Cp = size(Clabels,2);

if nargout<4
  idist = d;
  clear d;
  [idist,idx] = sort(idist,2);
else
  [idist,idx] = sort(d,2);
end
idist(idist==0) = realmin;
idist = 1./idist;

nPlabels = ones(size(Plabels));
for c=2:Cp
  nPlabels(Plabels==Clabels(c)) = c;
end
lab = nPlabels(idx);

A = zeros(Nx,K);
cnt = zeros(Nx,Cp);
dst = zeros(Nx,Cp);
for k=1:K
  labk = lab(:,k);
  for c=1:Cp
    sel = labk==c;
    cnt(sel,c) = cnt(sel,c)+1;
    dst(sel,c) = dst(sel,c)+idist(sel,k);
  end
  sel = cnt+dst./repmat(sum(dst,2),1,Cp);
  [labk,A(:,k)] = max(sel,[],2);
  if nargout>2
    if perclass
      S(:,:,k) = ((sel-0.5)./(k+1-0.5*repmat(sum(cnt~=0,2),1,Cp))).*(cnt~=0);
    else
      S(:,k) = (sel((A(:,k)-1)*Nx+[1:Nx]')-0.5)./(k+1-0.5*sum(cnt~=0,2));
    end
  end
end

if sum(size(Xlabels))~=0
  if perclass
    E = zeros(Cp,K);
    c = 1;
    for label=Clabels
      sel = Xlabels==label;
      for k=1:K
        E(c,k) = sum(Clabels(A(sel,k))~=label)/sum(sel);
      end
      c = c+1;
    end
  else
    E = zeros(1,K);
    for k=1:K
      E(k) = sum(Clabels(A(:,k)')~=Xlabels)/Nx;
    end
  end
end

if nargout>1
  A = Clabels(A);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%                            Helper functions                             %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% file: dstmat.m %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function d = classify_knn_dstmat( P, X, work, varargin )


fn = 'dstmat:';
minargs = 3;

if nargin==0
  d.dtype.euclidean = true;
  d.dtype.cosine = false;
  d.dtype.tangent = false;
  d.dtype.rtangent = false;
  d.dtype.otangent = false;
  d.dtype.atangent = false;
  d.dtype.hamming = false;
  return;
end

if ischar(P)
  return;
end

[ D, Np ] = size(P);
Nx = size(X,2);
d = [];

if isfield(work,'tangVp')
  tangVp = work.tangVp;
end
if isfield(work,'tangVx')
  tangVx = work.tangVx;
end

logfile = 2;

n = 1;
argerr = false;
while size(varargin,2)>0
  if ~ischar(varargin{n})
    argerr = true;
  elseif strcmp(varargin{n},'logfile') || ...
         strcmp(varargin{n},'tangVp') || ...
         strcmp(varargin{n},'tangVx')
    eval([varargin{n},'=varargin{n+1};']);
    if ~isnumeric(varargin{n+1})
      argerr = true;
    else
      n = n+2;
    end
  else
    argerr = true;
  end
  if argerr || n>size(varargin,2)
    break;
  end
end

if argerr
  fprintf(logfile,'%s error: incorrect input argument %d (%s,%g)\n',fn,n+minargs,varargin{n},varargin{n+1});
  return;
elseif nargin-size(varargin,2)~=minargs
  fprintf(logfile,'%s error: not enough input arguments\n',fn);
  return;
elseif Nx>0 && size(X,1)~=D
  fprintf(logfile,'%s error: dimensionality (rows) of both data matrices must be the same\n',fn);
  return;
elseif ~isfield(work,'dtype')
  fprintf(logfile,'%s error: dtype should be specified\n',fn);
  return;
elseif isfield(work.dtype,'tangent') && ~exist('tangVp','var') && ...
       ( work.dtype.tangent || work.dtype.atangent || work.dtype.rtangent )
  fprintf(logfile,'%s error: tangents of P should be given\n',fn);
  return;
elseif isfield(work.dtype,'tangent') && ~exist('tangVx','var') && Nx>0 && ...
       ( work.dtype.tangent || work.dtype.atangent || work.dtype.otangent )
  fprintf(logfile,'%s error: tangents of X should be given\n',fn);
  return;
elseif ( exist('tangVp','var') && mod(size(tangVp,2),Np)~=0 ) || ...
       ( exist('tangVx','var') && mod(size(tangVx,2),Nx)~=0 )
  fprintf(logfile,'%s error: number of tangents should be a multiple of the number of samples\n',fn);
  return;
end

dtype = work.dtype;
onesNp = ones(Np,1);
onesNx = ones(Nx,1);
onesD = ones(D,1);

cosnorm = true;
if isfield(work,'cosnorm')
  cosnorm = work.cosnorm;
end
normdist = false;
if isfield(work,'normdist')
  normdist = work.normdist;
end
torthonorm = false;
if isfield(work,'torthonorm')
  torthonorm = work.torthonorm;
end

if exist('tangVp','var') && ( dtype.rtangent || dtype.atangent || dtype.tangent )
  Lp = size(tangVp,2)/Np;
  if torthonorm || sum(sum(eye(Lp)-round(1000*tangVp(:,1:Lp)'*tangVp(:,1:Lp))./1000))~=0
    if ~torthonorm
      fprintf(logfile,'%s warning: tangVp not orthonormal, orthonormalizing ...\n',fn);
    end
    for nlp=1:Lp:size(tangVp,2)
      [ orthoVp, dummy ] = qr(tangVp(:,nlp:nlp+Lp-1),0);
      tangVp(:,nlp:nlp+Lp-1) = orthoVp;
    end
  end
end
if exist('tangVx','var') && ( dtype.otangent || dtype.atangent || dtype.tangent )
  Lx = size(tangVx,2)/Nx;
  if torthonorm || sum(sum(eye(Lx)-round(1000*tangVx(:,1:Lx)'*tangVx(:,1:Lx))./1000))~=0
    if ~torthonorm
      fprintf(logfile,'%s warning: tangVx not orthonormal, orthonormalizing ...\n',fn);
    end
    for nlx=1:Lx:size(tangVx,2)
      [ orthoVx, dummy ] = qr(tangVx(:,nlx:nlx+Lx-1),0);
      tangVx(:,nlx:nlx+Lx-1) = orthoVx;
    end
  end
end

if Nx>0

  % euclidean distance
  if dtype.euclidean
    x2 = sum((X.^2),1)';
    p2 = sum((P.^2),1);
    d = X'*P;
    d = x2(:,onesNp)+p2(onesNx,:)-d-d;
    if isfield(work,'sqrt') && work.sqrt
      d(d<0) = 0;
      d = sqrt(d);
    end
    if normdist
      d = (1/D).*d;
    end
  % cosine distance
  elseif dtype.cosine
    if cosnorm
      psd = sqrt(sum(P.*P,1));
      P = P./psd(onesD,:);
      xsd = sqrt(sum(X.*X,1));
      X = X./xsd(onesD,:);
    end
    if isfield(work,'cospos') && work.cospos
      d = 1-(X'*P+1)./2;
    else
      d = 1-X'*P;
    end
  % hamming distance
  elseif dtype.hamming
    lup = uint16([ ...
      0 1 1 2 1 2 2 3 1 2 2 3 2 3 3 4 1 2 2 3 2 3 ...
      3 4 2 3 3 4 3 4 4 5 1 2 2 3 2 3 3 4 2 3 3 4 ...
      3 4 4 5 2 3 3 4 3 4 4 5 3 4 4 5 4 5 5 6 1 2 ...
      2 3 2 3 3 4 2 3 3 4 3 4 4 5 2 3 3 4 3 4 4 5 ...
      3 4 4 5 4 5 5 6 2 3 3 4 3 4 4 5 3 4 4 5 4 5 ...
      5 6 3 4 4 5 4 5 5 6 4 5 5 6 5 6 6 7 1 2 2 3 ...
      2 3 3 4 2 3 3 4 3 4 4 5 2 3 3 4 3 4 4 5 3 4 ...
      4 5 4 5 5 6 2 3 3 4 3 4 4 5 3 4 4 5 4 5 5 6 ...
      3 4 4 5 4 5 5 6 4 5 5 6 5 6 6 7 2 3 3 4 3 4 ...
      4 5 3 4 4 5 4 5 5 6 3 4 4 5 4 5 5 6 4 5 5 6 ...
      5 6 6 7 3 4 4 5 4 5 5 6 4 5 5 6 5 6 6 7 4 5 ...
      5 6 5 6 6 7 5 6 6 7 6 7 7 8])';
    d = zeros(Nx,Np);
    for nx=1:Nx
      d(nx,:) = sum(lup(1+uint16(bitxor(P,X(:,nx(onesNp))))),1);
    end
    if normdist
      d = (1/(8*D)).*d;
    end
  % reference single sided tangent distance
  elseif dtype.rtangent
    d = zeros(Nx,Np);
    Lp = size(tangVp,2)/Np;
    nlp = 1;
    for np=1:Np
      dXP = X-P(:,np(onesNx));
      VdXP = tangVp(:,nlp:nlp+Lp-1)'*dXP;
      d(:,np) = (sum(dXP.*dXP,1)-sum(VdXP.*VdXP,1))';
      nlp = nlp+Lp;
    end
    if isfield(work,'sqrt') && work.sqrt
      d(d<0) = 0;
      d = sqrt(d);
    end
    if normdist
      d = (1/D).*d;
    end
  % observation single sided tangent distance
  elseif dtype.otangent
    d = zeros(Nx,Np);
    Lx = size(tangVx,2)/Nx;
    nlx = 1;
    for nx=1:Nx
      dXP = X(:,nx(onesNp))-P;
      VdXP = tangVx(:,nlx:nlx+Lx-1)'*dXP;
      d(nx,:) = sum(dXP.*dXP,1)-sum(VdXP.*VdXP,1);
      nlx = nlx+Lx;
    end
    if isfield(work,'sqrt') && work.sqrt
      d(d<0) = 0;
      d = sqrt(d);
    end
    if normdist
      d = (1/D).*d;
    end
  % average single sided tangent distance
  elseif dtype.atangent
    d = zeros(Nx,Np);
    Lp = size(tangVp,2)/Np;
    nlp = 1;
    for np=1:Np
      dXP = X-P(:,np(onesNx));
      VdXP = tangVp(:,nlp:nlp+Lp-1)'*dXP;
      d(:,np) = (sum(dXP.*dXP,1)-0.5*sum(VdXP.*VdXP,1))';
      nlp = nlp+Lp;
    end
    Lx = size(tangVx,2)/Nx;
    nlx = 1;
    for nx=1:Nx
      dXP = X(:,nx(onesNp))-P;
      VdXP = tangVx(:,nlx:nlx+Lx-1)'*dXP;
      d(nx,:) = d(nx,:)-0.5*sum(VdXP.*VdXP,1);
      nlx = nlx+Lx;
    end
    if isfield(work,'sqrt') && work.sqrt
      d(d<0) = 0;
      d = sqrt(d);
    end
    if normdist
      d = (1/D).*d;
    end
  % tangent distance
  elseif dtype.tangent
    d = zeros(Nx,Np);
    Lp = size(tangVp,2)/Np;
    Lx = size(tangVx,2)/Nx;
    tangVpp = zeros(Lp,Lp*Np);
    itangVpp = zeros(Lp,Lp*Np);
    tangVxx = zeros(Lx,Lx*Nx);
    itangVxx = zeros(Lx,Lx*Nx);
    nlp = 1;
    for np=1:Np
      sel = nlp:nlp+Lp-1;
      Vp = tangVp(:,sel);
      tangVpp(:,sel) = Vp'*Vp;
      itangVpp(:,sel) = inv(tangVpp(:,sel));
      nlp = nlp+Lp;
    end
    nlx = 1;
    for nx=1:Nx
      sel = nlx:nlx+Lx-1;
      Vx = tangVx(:,sel);
      tangVxx(:,sel) = Vx'*Vx;
      itangVxx(:,sel) = inv(tangVxx(:,sel));
      nlx = nlx+Lx;
    end
    nlx = 1;
    for nx=1:Nx
      sel = nlx:nlx+Lx-1;
      nlx = nlx+Lx;
      Vx = tangVx(:,sel);
      Vxx = tangVxx(:,sel);
      iVxx = itangVxx(:,sel);
      x = X(:,nx);
      nlp = 1;
      for np=1:Np
        sel = nlp:nlp+Lp-1;
        nlp = nlp+Lp;
        Vp = tangVp(:,sel);
        Vpp = tangVpp(:,sel);
        iVpp = itangVpp(:,sel);
        p = P(:,np);
        Vpx = Vp'*Vx;
        Alp = (Vpx*iVxx*Vx'-Vp')*(x-p);
        Arp = Vpx*iVxx*Vpx'-Vpp;
        Alx = (Vpx'*iVpp*Vp'-Vx')*(x-p);
        Arx = Vxx-Vpx'*iVpp*Vpx;
        ap = pinv(Arp)*Alp;
        ax = pinv(Arx)*Alx;
        xx = x+Vx*ax;
        pp = p+Vp*ap;
        d(nx,np) = (xx-pp)'*(xx-pp);
      end
    end
    if isfield(work,'sqrt') && work.sqrt
      d(d<0) = 0;
      d = sqrt(d);
    end
    if normdist
      d = (1/D).*d;
    end
  end

else

  % euclidean distance
  if dtype.euclidean
    p2 = sum((P.^2),1);
    d = P'*P;
    d = p2(onesNp,:)'+p2(onesNp,:)-d-d;
    if isfield(work,'sqrt') && work.sqrt
      d(d<0) = 0;
      d = sqrt(d);
    end
    if normdist
      d = (1/D).*d;
    end
  % cosine distance
  elseif dtype.cosine
    if cosnorm
      psd = sqrt(sum(P.*P,1));
      P = P./psd(onesD,:);
    end
    if isfield(work,'cospos') && work.cospos
      d = 1-(P'*P+1)./2;
    else
      d = 1-P'*P;
    end
  % hamming distance
  elseif dtype.hamming
    lup = uint16([ ...
      0 1 1 2 1 2 2 3 1 2 2 3 2 3 3 4 1 2 2 3 2 3 ...
      3 4 2 3 3 4 3 4 4 5 1 2 2 3 2 3 3 4 2 3 3 4 ...
      3 4 4 5 2 3 3 4 3 4 4 5 3 4 4 5 4 5 5 6 1 2 ...
      2 3 2 3 3 4 2 3 3 4 3 4 4 5 2 3 3 4 3 4 4 5 ...
      3 4 4 5 4 5 5 6 2 3 3 4 3 4 4 5 3 4 4 5 4 5 ...
      5 6 3 4 4 5 4 5 5 6 4 5 5 6 5 6 6 7 1 2 2 3 ...
      2 3 3 4 2 3 3 4 3 4 4 5 2 3 3 4 3 4 4 5 3 4 ...
      4 5 4 5 5 6 2 3 3 4 3 4 4 5 3 4 4 5 4 5 5 6 ...
      3 4 4 5 4 5 5 6 4 5 5 6 5 6 6 7 2 3 3 4 3 4 ...
      4 5 3 4 4 5 4 5 5 6 3 4 4 5 4 5 5 6 4 5 5 6 ...
      5 6 6 7 3 4 4 5 4 5 5 6 4 5 5 6 5 6 6 7 4 5 ...
      5 6 5 6 6 7 5 6 6 7 6 7 7 8])';
    d = zeros(Np,Np);
    for nx=1:Np
      d(nx,:) = sum(lup(1+uint16(bitxor(P,P(:,nx(onesNp))))),1);
    end
    if normdist
      d = (1/(8*D)).*d;
    end
  elseif dtype.rtangent || dtype.otangent || dtype.atangent || dtype.tangent
    fprintf(logfile,'%s error: not implemented\n',fn);
  end

end

if isfield(work,'nozero') && work.nozero
  d(d<eps) = eps;
elseif isfield(work,'noneg') && work.noneg
  d(d<0) = 0;
end
%%% file: dstmat.m %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
