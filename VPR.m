function VPR()
% all the proposals are valid, just sort them based on score and get the top
% 100.

% Ref: http://eprints.qut.edu.au/84931/1/rss15_placeRec.pdf
tic;
%% -------------------- START_UP -------------------- 
% !ldd ./matlab/+caffe/private/caffe_.mexa64 
addpath('/home/sepideh/Documents/cmput617-project/edges/private');

addpath(genpath('.'));
close all;
clc;
clear;
clear mex;
caffe.reset_all();

caffe.set_mode_gpu();
gpu_id = 0;  % we will use the first gpu in this demo
caffe.set_device(gpu_id);

%% ------------------- INIT ----------------------

% Edge_boxes

% load pre-trained edge detection model and set opts (see edgesDemo.m)
edge_model=load('edges/models/forest/modelBsds'); edge_model=edge_model.model;
edge_model.opts.multiscale=0; edge_model.opts.sharpen=2; edge_model.opts.nThreads=4;

% ConvNet

% model 1
cnnmodel = './caffe/models/bvlc_alexnet/trunc_deploy.prototxt';
cnnweights = './caffe/models/bvlc_alexnet/bvlc_alexnet.caffemodel';
% model 2
% cnnmodel = './caffe/models/bvlc_reference_rcnn_ilsvrc13/trunc_deploy.prototxt';
% cnnweights = './caffe/models/bvlc_reference_rcnn_ilsvrc13/bvlc_reference_rcnn_ilsvrc13.caffemodel';
% model 3
% cnnmodel = './deep-residual-networks/prototxt/trunc_ResNet-50-deploy.prototxt';
% cnnweights = './deep-residual-networks/prototxt/ResNet-50-model.caffemodel';
d = load('./caffe/matlab/+caffe/imagenet/ilsvrc_2012_mean.mat');

mean_data = imresize(d.mean_data, [231 231], 'bilinear');

net = caffe.Net(cnnmodel, cnnweights, 'test'); % create net and load weights
net.blobs('data').reshape([231 231 3 1]); % reshape blob 'data'
net.reshape();

%% ---------------------- IMAGE MATCHING ----------------------
imgType = '*.jpg'; 

seen_imgPath = './small_dataset/day_right';
seen_images  = dir([seen_imgPath '/' imgType]);
[MatFeat0, MatBBox0] = get_features(seen_images, seen_imgPath, edge_model, net, mean_data,'seenIm');

test_imgPath = './small_dataset/night_right';
test_images  = dir([test_imgPath '/' imgType]);
[MatFeat1, MatBBox1] = get_features(test_images, test_imgPath, edge_model, net, mean_data ,'testIm');

similarity_Mat = getSimilarityMat(MatFeat0, MatBBox0, MatFeat1, MatBBox1, length(seen_images), length(test_images)); 
comp_precision_recall(similarity_Mat);

toc;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%                              Functions                                  %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [MatFeat, MatBBox] = get_features(images, imgPath, edge_model, net, mean_data, kindIm)

for j = 1:length(images)
    
    im = imread([imgPath '/' images(j).name]);
    bboxes=edgeBoxes(im,edge_model);
    [MatFeat{j}, MatBBox{j}] = fearures_box(bboxes, im, net, mean_data, j, kindIm); 
    
end

end

function matrixD = getSimilarityMat(MatFeat0, MatBBox0, MatFeat1, MatBBox1, numIm0, numIm1)
% Computing score for Ia, Ib

% Input:
% 	MatFeat0 - the feature matrix of loop0
% 	MatBBox0 - the bounding box matrix of loop0
% 	MatFeat1 - the feature matrix of loop1
% 	MatBBox1 - the bounding box matrix of loop1 
% 
% Output:
% 	matrixD - the similarity matrix of two loops
%     


for idxIm0 = 1:numIm0
	imFeat0 = MatFeat0{idxIm0};
	bbox0 = MatBBox0{idxIm0};
	
	numProposals = size(imFeat0, 1);
	for idxIm1 = 1:numIm1
		imFeat1 = MatFeat1{idxIm1};
		bbox1 = MatBBox1{idxIm1};
		d01 = pdist2(imFeat0, imFeat1, 'cosine');
		
		[maxVal01 maxIdx01] = min(d01, [], 2);
		[maxVal10 maxIdx10] = min(d01, [], 1);
		
		sumSimilarity = 0.0;
		for ii = 1:numProposals
			maxVal01_tmp = maxVal01(ii, 1);
			maxIdx01_tmp = maxIdx01(ii, 1);
			maxIdx10_tmp = maxIdx10(1, maxIdx01_tmp);
			
			if ii == maxIdx10_tmp
				w0 = bbox0(ii, 3) %- bbox0(ii, 1);
				h0 = bbox0(ii, 4) %- bbox0(ii, 2);
				w1 = bbox1(maxIdx01_tmp, 3) %- bbox1(maxIdx01_tmp, 1);
				h1 = bbox1(maxIdx01_tmp, 4) %- bbox1(maxIdx01_tmp, 2);
				maxVal01_tmp = maxVal01_tmp * exp( (abs(w0 - w1)*1.0/max(w0, w1) + abs(h0 - h1)*1.0/max(h0, h1)) / 2.0 );
				sumSimilarity = sumSimilarity + (1 - maxVal01_tmp);
			end
		end
		
		% get the similarity of two images
		matrixD(idxIm0, idxIm1) = sumSimilarity;
	end
end

end

function comp_precision_recall(similarity_Mat)

for thre = 0:0.1:1 % similarity threshold
    S=similarity_Mat;
    A1 = find(S>thre);A0 = find(S<thre);
    S(A1) = 1; S(A0) = 0;
    for i=1:size(S,1)
        for j=1:size(S,2)
            if i==j
               TP=TP+(S(i,j)==1);
               FN=FN+(S(i,j)==0);
            else
               FP=FP+(S(i,j)==1);
            end
        end
    end
    
    [score, AssignIdx] = max(S);
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

function V = convertToVec(M)
% convertToVec is converting the nxpxq matrix M to a vector V

[n,p,q] = size(M);
V = [];
for j = 1:p
    for k = 1:q
        V = [V; M(:,j,k)];
    end
end
end

function [MatFeat, MatBBox] = fearures_box(bboxes, im, net, mean_data, im_id, kindIm)
% for each box of the image, calculate the feature and project it.

MatFeat = []; % each row is a feature vector for a box

[d1, d2, d3] = size(im);

[bul, I] = sort(bboxes(:,end),'descend');
bboxes = bboxes(I,:);

nLandmarks = min(100, size(bboxes,1));
bboxes = bboxes(1:nLandmarks, :);
MatBBox = bboxes;

mkdir(sprintf('./test_images/%s/%s',kindIm,num2str(im_id)));

box_size=231;

for i = 1:nLandmarks

    x=bboxes(i, 1); y=bboxes(i, 2); h=bboxes(i, 3); w=bboxes(i, 4);
    
    bounding_box = im(x:x+box_size-1, y:y+box_size-1, :); 
    
    imwrite(bounding_box,['./test_images/',kindIm,'/',num2str(im_id),'/',num2str(i),'.png']);

    % Convert an image returned by Matlab's imread to im_data in caffe's data
    im_data = bounding_box(:, :, [3, 2, 1]);  % permute channels from RGB to BGR
    im_data = permute(im_data, [2, 1, 3]);  % flip width and height
    im_data = single(im_data);  % convert from uint8 to single
    im_data = imresize(im_data, [box_size box_size], 'bilinear');  % resize im_data
    im_data = im_data - mean_data;  % subtract mean_data (already in W x H x C, BGR)
    
    res = net.forward({im_data});   
    feature = res{1}; % get feature 
    feature = convertToVec(feature);
    
    % Gaussian Random Projection of the feature 
    lambda = 0.4;
    d = 512; % the dimension which is projected to.
    perf = regressiontest(feature',[],[],[],'gaussian','rp_factorize_large_real',lambda,d);
    feature = perf.u; % Transfered Features [1024x1]
    
    MatFeat = [MatFeat, feature];
    
    % caffe.reset_all();

end

end








