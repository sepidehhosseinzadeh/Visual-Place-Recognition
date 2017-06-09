function [G, W] = rpbin(X,nparts,kernel)

G = spalloc(0,0,0);

for p = 1:nparts
  [partition, W{p}.epsilon, W{p}.u, W{p}.parttab] = l1part(X',kernel);
  W{p}.nbins = max(partition);
  G = [G; part2G(partition)];
  fprintf('%d(%d) ', p, W{p}.nbins);
end
end


function [partition, epsilons, u, parttab] = l1part(X,kernel)
D = size(X,2);
switch kernel
 case 'laplacian'
   epsilons = gamrnd(2,1,[D,1]);
 otherwise
  error('kernel type not yet supported');
end
u = rand(D,1).*epsilons;

[partition,parttab] = binning(X,epsilons,u);
end
