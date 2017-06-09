function J = preProcess(I)
    lvl = graythresh(I);
    J   = im2bw(I,lvl);
end

