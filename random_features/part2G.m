function G = part2G(part,maxpart)
if nargin==1
  maxpart = max(part);
end
N = length(part);
i = find(part>=0);
G = sparse(double(part(i)),i,logical(1),double(maxpart),N,N);
end
