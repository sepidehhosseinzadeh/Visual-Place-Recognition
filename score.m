function [AssignIdx S] = score()
% Computing score for Ia, Ib

clc;
clear;
close all;
% load ('dataForScore_AlexNet.mat','n_seen','n_test','X','X_id','X_n','X_whs','Y','Y_id','Y_n','Y_whs');
load matlab.mat

K = 30; % knn
precisionV = [];
recallV = [];
n = size(X,2); % #of landmarks in seen images
m = size(Y,2); % #of landmarks in test images

Dx = pdist2(X',Y','cosine');
D = Dx-0.9;
Dy = pdist2(Y',X','cosine');

[Dx, Ix] = sort(Dx');
[Dy, Iy] = sort(Dy');
Ix = Ix';
Iy = Iy';
Ix = Ix(1:K, :);
Iy = Iy(1:K, :);


S = zeros(n_seen, n_test); % computing score for Ia and Ib
s = zeros(n,m);
FP = 0; FN = 0; TP = 0; 
for i=1:n
    for j=1:m

        i_id = X_id(i);
        j_id = Y_id(j);

        wi = X_whs(1,i); wj = Y_whs(1,j);
        hi = X_whs(2,i); hj = Y_whs(2,j);
        s(i,j) = exp(0.5 * ( abs(wi-wj)/max(wi,wj) + abs(hi-hj)/max(hi,hj) ));

        if ~isempty(find(Ix(:,j) == i)) && ~isempty(find(Iy(:,i) == j)) 

            Sij = 1/( sqrt(X_n(i_id))*sqrt(Y_n(j_id)) ) * (1 - D(i,j) .* s(i,j));
            S(i_id,j_id) = S(i_id,j_id) + Sij; 
        end
    end
end

% [score, AssignIdx] = max(S);
% precision = sum(AssignIdx == 1:length(AssignIdx))/length(AssignIdx)

for thre = 0:0.1:1 % similarity threshold
    SS=S;
    A1 = find(SS>thre);A0 = find(SS<thre);
    SS(A1) = 1; SS(A0) = 0;
    for i=1:n_seen
        for j=1:n_test
            if i==j
               TP=TP+(SS(i,j)==1);
               FN=FN+(SS(i,j)==0);
            else
               FP=FP+(SS(i,j)==1);
            end
        end
    end
    
    [score, AssignIdx] = max(SS);
    thre
    precision = TP / (TP+FP) 
    recall = TP / (TP+FN) 

    precisionV = [precisionV, precision];
    recallV = [recallV, recall];
end

plot(recallV,precisionV);
xlabel('recall');
ylabel('precision');
title('precision-recall graph');

end
    