% % transferlearn
% % written in March 2025
% % CLPS 0950: Intro to Programming
% % GitHub token: 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

% folderName = "C:\Users\drake\MATLAB\Projects\PunchNetDataset\PunchData\PunchData";
% imds = imageDatastore(folderName, ...
%     IncludeSubfolders=true, ...
%     LabelSource="foldernames");

% Splits data into training, validation, and testing pools
imageAugmenter = imageDataAugmenter( ...
    'RandRotation',[-20,20], ...
    'RandXTranslation',[-10 10], ...
    'RandYTranslation',[-10 10], ...
    'RandXReflection',true);

[imdsTrain,imdsValid,imdsTest] = splitEachLabel(imds,0.90,0.07,0.03,"randomized");
augimdsTrain = augmentedImageDatastore([224 224], imdsTrain, "ColorPreprocessing","gray2rgb", 'DataAugmentation', imageAugmenter);
augimdsValid = augmentedImageDatastore([224 224], imdsValid, "ColorPreprocessing","gray2rgb");
augimdsTest = augmentedImageDatastore([224 224], imdsTest, "ColorPreprocessing","gray2rgb");

% Displays Sample Images
numImages = numel(augimdsTrain.Files);
idx = randperm(numImages,16);
I = imtile(augimdsTrain.Files,Frames=idx);
figure
imshow(I)
%%
% trainset = zeros(224, 224, 3, numel(augimdsTrain.Files), 'uint8');
% for i = 1:numel(augimdsTrain.Files)
%     img = readimage(augimdsTrain, i);
%     if size(img, 3) == 1
%         img = repmat(img, [1 1 3]); % Convert grayscale to RGB
%     end
%     trainset(:, :, :, i) = imresize(img, [224 224]);
% end

% validset = zeros(224, 224, 3, numel(imdsValid.Files), 'uint8');
% for i = 1:numel(imdsValid.Files)
%     img = readimage(imdsValid, i);
%     if size(img, 3) == 1
%         img = repmat(img, [1 1 3]); % Convert grayscale to RGB
%     end
%     validset(:, :, :, i) = imresize(img, [224 224]);
% end
%%
augimdsTrain.MiniBatchSize = 32;
load net_1.mat
options = trainingOptions("sgdm", ...
    ValidationData=augimdsValid, ...
    ValidationFrequency=5, ...
    Plots="training-progress", ...
    Metrics="accuracy", ...
    Verbose=false, ...
    ExecutionEnvironment = 'gpu', ...
    MiniBatchSize=32, ...
    MaxEpochs=10);
net = trainnet(augimdsTrain, trainednet3,"crossentropy",options);

%%

load trainednet1.mat
load trainednet2.mat

classNames = categories(imds.Labels);

YTest1 = minibatchpredict(trainednet1,augimdsTest);
YTest1 = scores2label(YTest1,classNames);
YTest2 = minibatchpredict(net,augimdsTest);
YTest2 = scores2label(YTest2,classNames);

TTest = imdsTest.Labels;
figure
subplot(1, 2, 1);
title('trainednet1');
confusionchart(TTest,YTest1);
subplot(1, 2, 2);
title('net');
confusionchart(TTest,YTest2);
%%
im = imread("C:\Users\drake\MATLAB\Projects\PunchNetData\PunchData\PunchData\Hit\H (57).jpg");

im = imresize(im,[224 224]);
X = single(im);

scores1 = predict(trainednet3,X);
scores2 = predict(net,X);
[label1,score1] = scores2label(scores1,classNames);
[label2,score2] = scores2label(scores2,classNames);

figure
imshow(im)
title(['1: ' string(label1) + " (Score: " + gather(score1) + ")" ...
    '2: ' string(label2) + " (Score: " + gather(score2) + ")"])

