% Applies the regressor defined in perf to testing input points Xtest,
% and compares the result with ytest. Perf is the output of REGRESSIONTEST.
% Tries to be smart about how it reports the prediction error.
%
function perf=  evalregression(perf,Xtest,ytest)
fprintf('evaluating...\n');
tic;
yhat = zeros(1,size(Xtest,2));
blk = 300;
for i=1:blk:size(Xtest,2)
  en = min(i+blk-1,size(Xtest,2));
  switch perf.method
   case {'exact','nystrom'}
    Ktest = feval(getfield(perf.kernels,perf.kernel),Xtrain,Xtest(:,i:en));
    yhat(i:en) = c'*Ktest;
   case {'rp_factorize','rp_factorize_large'}
    Gtest = rp_apply(Xtest(:,i:en),perf.W(:,i:i+size(Xtest,1)-1));
    yhat(i:en) = real(perf.u'*Gtest);
   case 'rpbin'
    Gtest = rpbin_apply(Xtest(:,i:en),perf.W);
    yhat(i:en) = real(perf.u'*Gtest);
   case 'rp_factorize_large_real'
    Gtest = rp_apply_real(Xtest(:,i:en),perf.W);
    yhat(i:en) = perf.u'*Gtest;
  end
end

yhat = yhat+perf.ytrain_mean;
perf.yhat = yhat;
perf.evaltime = toc;


fprintf('done.\n');

end


function er = error_report(ytest,yhat,kernel,method,lambda,varargin)
classification = is_classification(ytest);

if classification
    er = 0.5-ytest(:)'*sign(yhat(:))/length(ytest(:))/2;
    cls = 'CLASSIFICATION'
else
    er = norm(ytest(:)-yhat(:))/norm(ytest(:));
    cls = 'REGRESSION'
end
fprintf('error %g\n', er);
plot(1:length(ytest), ytest,'k.-',1:length(yhat),yhat,'x-');
switch method
    case {'rp_factorize','nystrom'}
        d = varargin{1};
        title(sprintf('%s %s %s \\lambda=%g d=%d err=%g', cls, quotename(method), kernel,lambda,d, er));
    otherwise
        title(sprintf('%s %s %s \\lambda=%g err=%g', cls, quotename(method), kernel,lambda, er));
end
legend('true',kernel);
end

function b = is_classification(y)
b = all(y==-1 | y==1);
end


function str = quotename(str)
str = ['"' strrep(str,'_',' ') '"'];
end
