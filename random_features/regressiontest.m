% Benchmarks Gaussian Process Regression (Krigging) using Random Features,
% exact, and Nystrom methods.
%
% This is research code used to generate the results that appear in
%
% Random Features for Large-Scale Kernel Machines, Ali Rahimi, Ben Recht,
% to appear in Neural Information Processing Systems (NIPS) 2007.
%
% Xtrain: dxN matrix of N d-dimensional training inputs
% ytrain: Nx1 vector N scalar training outputs
% Xtest: optional testing input vector (you may pass [])
% ytest: optional testing output vector (you may pass [])
% kernel: the string 'gaussian', 'laplacian', or 'linear' is the Kernel whose
%  random features are to be generated.
% method: 'calibrvar' returns a good guess for what kernel variance to use.
%         'exact' solve the exact least squares problem
%         'nystrom' use the Nystrom approximation
%         'nystrom' use the Nystrom approximation
%         'rp_factorize' Fourier Random Features
%         'rp_factorize_large' Fourier Random Features for problems that don't
%              fit in memory (the code may be outdated)
%         'rp_factorize_large_real' Cosine Random Features for problems that
%              don't fit in memory  (the code may be outdated)
%         'rp_bin' Random Binning Features
%
%
% Returns a structure that can be used to evaluate the regressor at
% other points using the EVALREGRESSION function.



function perf = regressiontest(Xtrain,ytrain,Xtest,ytest,kernel,method,lambda,varargin)

N = size(Xtrain,2);

kernels = struct('gaussian',@kernel_gaussian,...
                 'laplacian',@kernel_laplacian,...
                 'linear',@kernel_linear);

ytrain_mean = mean(ytrain);
ytrain = ytrain-ytrain_mean;

perf.method = method;
perf.lambda = lambda;
%perf.kernels = kernels;
perf.kernel = kernel;
perf.ytrain_mean = ytrain_mean;

fprintf('Factoring...\n');
tic
switch method
 case 'calibvar'
  perf = calibvar(Xtrain); return;
 case 'exact'
  K = feval(getfield(kernels,kernel),Xtrain);
 case 'nystrom'
  d = varargin{1};
  [G,W] = nystrom(Xtrain,d,getfield(kernels,kernel));
 case 'rp_factorize'
  d = varargin{1};
  [G,W] = rp_factorize(Xtrain,d,kernel);
 case 'rp_factorize_large'
  d = varargin{1};
  [GG,Gy,W] = rp_factorize_large(Xtrain,ytrain,d,kernel,2000);
 case 'rp_factorize_large_real'
  d = varargin{1};
  [GG,Gy,W] = rp_factorize_large_real(Xtrain,Xtrain,d,kernel,1000);
 case 'rp_bin'
  d = varargin{1};
  [G,W] = rpbin(Xtrain,d,kernel);
  spy(G);drawnow
 otherwise
  error('Don''t have a test like that');
end
perf.factorizetime = toc;



if exist('d','var')
  perf.rps = d;
end
if exist('W','var')
  perf.W = W;
end

fprintf('solving...\n');
tic;
switch method
 case 'exact'
  c = (K+eye(N)*lambda)\ytrain(:);
  perf.c = c;
 case 'nystrom'
  c = lowranksolver(G,ytrain(:),lambda);
  perf.c = c;
 case {'rp_factorize','rpbin'}
  u = lowranksolver2(G,Xtrain(:),d*lambda);
  perf.u = u;
 case {'rp_factorize_large','rp_factorize_large_real'}
  u = lowranksolver3(GG,Gy,d*lambda);
  perf.u = u;
end
perf.solvetime = toc;

% if ~isempty(Xtest)
%   perf = evalregression(perf,Xtest,ytest);
% end
end


function v = calibvar(X)
p = logical(binornd(1,200/size(X,2),1,size(X,2)));
R= L2_distance(X(:,p),X(:,p)).^2;
v= mean(R(:));
end


function K = kernel_linear(X,Y)
if nargin==1
    Y = X;
end
K = X'*Y;
end

function K = kernel_gaussian(X,Y)
if nargin==1
    Y = X;
end
R = L2_distance(X,Y);
K = exp(-R.^2);
end

function K = kernel_laplacian(varargin)
R = L1_distance(varargin{:});
K = exp(-R);
end
