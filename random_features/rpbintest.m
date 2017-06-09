D = 8;
N = 4000;
Ntest = 100;
RP = 100;

lambda = 0.1;

X = (0.08)*randn(D,N);
Xtest = (0.08)*randn(D,Ntest);
A = randn(1,8);
y = A*X;
ytest = A*Xtest;


[G,W] = rpbin(X,RP,'laplacian');
Gtest = rpbin_apply(Xtest,W);
u = lowranksolver2(G,y(:),RP*lambda);
yhat = u'*Gtest;
plot(1:Ntest, yhat, '.-', 1:Ntest, ytest,'k--')

%c = lowranksolver(G,y(:),RP*lambda);
%yhat = Gtest'*(G*c);
%plot(1:Ntest, yhat, '.-', 1:Ntest,yhat3, 'g-', 1:Ntest, ytest,'k--')


return
K = exp(-L1_distance(X));
c = (K+lambda*eye(N))\y(:);
Ktest = exp(-L1_distance(X,Xtest));
yhat2 = c'*Ktest;
plot(1:100, yhat2, '.-', 1:100, ytest,'k--')
