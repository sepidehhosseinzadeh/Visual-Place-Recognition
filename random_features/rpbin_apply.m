function G = rpbin_apply(Xtest,W)
nparts = length(W);
G = spalloc(0,0,0);
for p=1:nparts
  partition = binning(Xtest',W{p}.epsilon, W{p}.u,W{p}.parttab);
  G = [G; part2G(partition, W{p}.nbins)];
end
end
