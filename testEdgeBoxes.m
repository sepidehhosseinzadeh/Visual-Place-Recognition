addpath(genpath('.'));
close all;
clc;
clear;
clear mex;

caffe.set_mode_gpu();
gpu_id = 0;  % we will use the first gpu in this demo
caffe.set_device(gpu_id);

caffe.reset_all();

% Edge_boxes

% load pre-trained edge detection model and set opts (see edgesDemo.m)
edge_model=load('edges/models/forest/modelBsds'); edge_model=edge_model.model;
edge_model.opts.multiscale=0; edge_model.opts.sharpen=2; edge_model.opts.nThreads=4;

%% -------------------- TESTING --------------------
imgType = '*.jpg'; 
imgPath = './small_dataset/day_right';
images  = dir([imgPath '/' imgType]);

for j = 1:length(images)
    mkdir(['test' '/' int2str(j)])
    im = imread([imgPath '/' images(j).name]);

    bboxes=edgeBoxes(im,edge_model);
    gt = bboxes;
    [bul, I] = sort(gt(:,end),'descend');
    gt = gt(I,:);
    nLandmarks = min(100, size(gt,1));
    gt = gt(1:nLandmarks, :);
    [gtRes,dtRes]=bbGt('evalRes',gt,double(bboxes),.7);
    
    for i = 1:size(gtRes,1)
        box = gtRes(i, 1:4);
        x=box(1); y=box(2); h=box(3)-x; w=box(4)-y;
        imwrite(im(x:x+h-1, y:y+w-1, :), ['test1' '/' int2str(j) '/' images(j).name '_' int2str(i) '.jpg']);
    end
    
end

caffe.reset_all();

