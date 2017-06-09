% diary
% 
X_train = loadMNISTImages('train-images.idx3-ubyte'); %images
Y_train = loadMNISTLabels('train-labels.idx1-ubyte'); %labels
Y_train = Y_train';

[idxCol, distance] = vl_kdtreequery(vl_kdtreebuild(X_train), X_train, X_train(:,1), 'NumNeighbors', 10000);
X_train = X_train(:,idxCol);
Y_train = Y_train(idxCol);

X_test = loadMNISTImages('t10k-images.idx3-ubyte'); %images
Y_test = loadMNISTLabels('t10k-labels.idx1-ubyte'); %labels
Y_test = Y_test';

[idxCol, distance] = vl_kdtreequery(vl_kdtreebuild(X_test), X_test, X_test(:,1), 'NumNeighbors', 10000);
X_test = X_test(:,idxCol);
Y_test = Y_test(idxCol);


 err = zeros(1,10); % computing errors for k=1:10
 for k=2:10
     reorder = randperm(size(X_train,2));
     Xr = X_train(:,reorder);
     Yr = Y_train(:,reorder);
     sizeFold = size(Xr,2)/10;
 
     error = 0;
     for i=1:10 % 10 fold cross-validation
         Xf_train = removerows(Xr',(i-1)*sizeFold+1:i*sizeFold);
         Xf_train = Xf_train';
         Yf_train = removerows(Yr',(i-1)*sizeFold+1:i*sizeFold);
         Yf_train = Yf_train';
         Xf_test = Xr(:,((i-1)*sizeFold)+1:i*sizeFold);
         Yf_test = Yr(((i-1)*sizeFold)+1:i*sizeFold);
 
         [tangVp, Vidx] = tangVects(Xf_train,strcat('k',num2str(k)),'basis',true,'Xlabels',Yf_train,'knnprotos',Xf_train,Yf_train);
         [tangVx, Vidy] = tangVects(Xf_test,strcat('k',num2str(k)),'basis',true,'Xlabels',Yf_test,'knnprotos',Xf_test,Yf_test);
         
         if size(tangVp,1)==0 || size(tangVx,1)==0
             continue;
         end
         
         [ E, A, S, d ] = classify_knn( Xf_train, Yf_train, k, Xf_test, Yf_test, 'tangent','tangVp',tangVp,'tangVx',tangVx);
         error = error+sum(E)/length(E);
         i
     end
     k
     err(k) = error/10;
 end
 
 [minValErr bestK] = min(err);
 minValErr
 bestK
%% run for best k = 4
[tangVp, Vidx] = tangVects(X_train,'k4','basis',true,'Xlabels',Y_train,'knnprotos',X_train,Y_train);
[tangVx, Vidy] = tangVects(X_test,'k4','basis',true,'Xlabels',Y_test,'knnprotos',X_test,Y_test);
        
[ E, A, S, d ] = classify_knn(X_train, Y_train, 4, X_test, Y_test, 'tangent','tangVp',tangVp,'tangVx',tangVx);
E       

