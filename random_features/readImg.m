function im = readImg(M, i)

m = size(M,2);
imgSize = sqrt(m);

im = reshape(M(i,:), imgSize, imgSize);

end

