function [partition, parttab] = findpartition(I, parttab)
D = size(I,2);
partition = zeros(size(I,1),1,'int32');
goods = logical(ones(1,size(I,1),'int8') );

for d=1:D
  if nargin>1
    p = recall(parttab{d},I(:,d));
    goods = goods & (p>=0);
    partition = partition*(1+max(parttab{d}(:,2))) + p(:);
  else
    [p, parttab{d}] = part1D(I(:,d));
    parttab{d} = parttab{d}';
    partition = partition*(1+max(p)) + p(:);
  end
end

if nargin>1
  p(goods) = recall(parttab{D+1},partition(goods));
  goods(goods) = goods(goods) & (p(goods)>=0);
else
  [p, parttab{D+1}] = part1D(partition);
  parttab{D+1} = parttab{D+1}';
end
partition = 1+p;

%if any(~goods)
%  error('bad');
%end

if nargin==1 && any(~goods)
  error('fucked up');
end
if nargin>1
  partition(~goods) = -1;
end
end


function y = recall(tab,x)
y = recall_mex(tab,x);
end
