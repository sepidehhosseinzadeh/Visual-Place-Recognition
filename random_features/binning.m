function [partition,parttab] = binning(X,epsilons,u,parttab)
N = size(X,1);
D = size(X,2);
Xdisc = zeros(N,D,'int32');
for d=1:D
  Xdisc(:,d) = discretize(X(:,d),u(d),epsilons(d));
end
if nargin>3
  partition = findpartition(int32(Xdisc),parttab);
else
  [partition, parttab] = findpartition(int32(Xdisc));
end
end


function Xdisc = discretize(x,u,e)
Xdisc = int32(round((x-u)/e));
end
