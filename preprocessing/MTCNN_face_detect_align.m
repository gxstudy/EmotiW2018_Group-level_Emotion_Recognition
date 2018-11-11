function face_detect_align()

clear;clc;close all;

%% collect a image list of CASIA & LFW
%trainList = collectData(fullfile(pwd, 'data/CASIA-WebFace'), 'CASIA-WebFace');
trainList = collectData('../data/test', 'test_faces_MTCNN');
%testList  = collectData(fullfile(pwd, 'data/lfw'), 'lfw');

%% mtcnn settings
minSize   = 20;
%factor    = 0.85;
factor = 0.709;
threshold = [0.6 0.7 0.7];%[0.6 0.7 0.9];

%% add toolbox paths
%matCaffe       = fullfile(pwd, '../tools/caffe-sphereface/matlab');
matCaffe='/home/xin/caffe_matlab/matlab';
%pdollarToolbox = fullfile(pwd, '../tools/toolbox');
pdollarToolbox='./MTCNN/MTCNN_face_detection_alignment-master/code/codes/toolbox-master';
%MTCNN          = fullfile(pwd, '../tools/MTCNN_face_detection_alignment/code/codes/MTCNNv1');
MTCNN = './MTCNN/MTCNN_face_detection_alignment-master/code/codes/MTCNNv2';
addpath(genpath(matCaffe));
addpath(genpath(pdollarToolbox));
addpath(genpath(MTCNN));

%% caffe settings
gpu = 1;
if gpu
   gpu_id = 0;
   caffe.set_mode_gpu();
   caffe.set_device(gpu_id);
else
   caffe.set_mode_cpu();
end
caffe.reset_all();
%modelPath = fullfile(pwd, '../tools/MTCNN_face_detection_alignment/code/codes/MTCNNv1/model');
modelPath = './MTCNN/MTCNN_face_detection_alignment-master/code/codes/MTCNNv2/model';
PNet = caffe.Net(fullfile(modelPath, 'det1.prototxt'), ...
                 fullfile(modelPath, 'det1.caffemodel'), 'test');
RNet = caffe.Net(fullfile(modelPath, 'det2.prototxt'), ...
                 fullfile(modelPath, 'det2.caffemodel'), 'test');
ONet = caffe.Net(fullfile(modelPath, 'det3.prototxt'), ...
                 fullfile(modelPath, 'det3.caffemodel'), 'test');
LNet  = caffe.Net(fullfile(modelPath, 'det4.prototxt'), ...
                 fullfile(modelPath, 'det4.caffemodel'), 'test');

%% face and facial landmark detection
%dataList = [trainList; testList];
dataList = trainList;

imgSize     = [112, 96];
coord5point = [30.2946, 51.6963;
               65.5318, 51.5014;
               48.0252, 71.7366;
               33.5493, 92.3655;
               62.7299, 92.2041];
           
for i = 1:length(dataList)
    fprintf(' detecting the %dth image...\n', i);
    fprintf(dataList(i).file);
    % load image
    img = imread(dataList(i).file);
    if size(img, 3)==1
       fprintf('gray image')
       img = repmat(img, [1,1,3]);
    end
    % detection
    [bboxes, landmarks] = detect_face(img, minSize, PNet, RNet, ONet, LNet,threshold, false, factor);

    if size(bboxes, 1)>=1
       for face_index = 1:size(bboxes,1)
           transf   = cp2tform(double(reshape(landmarks(:,face_index),[5,2])), coord5point, 'similarity');
           cropImg  = imtransform(img, transf, 'XData', [1 imgSize(2)],...
                                        'YData', [1 imgSize(1)], 'Size', imgSize);
    % save image
           [sPathStr, name, ext] = fileparts(dataList(i).file);
           %tPathStr = strrep(sPathStr, '/test/', '/faces_MTCNN_test/');
           tPathStr = fullfile('../data/test_faces_MTCNN', name);
           if ~exist(tPathStr, 'dir')
              mkdir(tPathStr)
           end
           imwrite(cropImg, fullfile(tPathStr, ['M_', num2str(round(bboxes(face_index,1))), '_', num2str(round(bboxes(face_index,2))),'_', num2str(round(bboxes(face_index,3)-bboxes(face_index,1))), '_', num2str(round(bboxes(face_index,4)-bboxes(face_index,2))) '.jpg']), 'jpg');
       % pick the face closed to the center
           %center   = size(img) / 2;
           %distance = sum(bsxfun(@minus, [mean(bboxes(:, [2, 4]), 2), ...
                                      %mean(bboxes(:, [1, 3]), 2)], center(1:2)).^2, 2);
           %[~, Ix]  = min(distance);
           %dataList(i).facial5point = reshape(landmarks(:, Ix), [5, 2]);
       end
    end
    delete(dataList(i).file);
end
%save dataList.mat dataList

end


function list = collectData(folder, name)
%     subFolders = struct2cell(dir(fullfile(folder, '*.jpg')))';
%     files      = fullfile(folder, subFolders(:, 1));
%     dataset    = cell(size(files));
%     dataset(:) = {name};
%     list       = cell2struct([files dataset], {'file', 'dataset'}, 2);

   
    subList  = struct2cell(dir(fullfile(folder, '*.jpg')))';
    files = fullfile(folder, subList(:, 1));

    dataset    = cell(size(files));
    dataset(:) = {name};
    list       = cell2struct([files dataset], {'file', 'dataset'}, 2);
end
