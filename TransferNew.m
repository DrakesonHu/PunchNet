% % transferlearn
% % written in March 2025
% % CLPS 0950: Intro to Programming
% % GitHub token: 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

folderName = "PunchDataCombined";
imds = imageDatastore(folderName, ...
    IncludeSubfolders=true, ...
    LabelSource="foldernames");

% Splits data into training, validation, and testing pools
imageAugmenter = imageDataAugmenter( ...
    'RandRotation',[-20,20], ...
    'RandXTranslation',[-10 10], ...
    'RandYTranslation',[-10 10], ...
    'RandXReflection',true);

[imdsTrain,imdsValid,imdsTest] = splitEachLabel(imds,0.85,0.10,0.05,"randomized");
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
load net_2.mat
options = trainingOptions("sgdm", ...
    ValidationData=augimdsValid, ...
    ValidationFrequency=5, ...
    LearnRateSchedule="cosine", ...
    Plots="training-progress", ...
    Metrics="accuracy", ...
    Verbose=false, ...
    ExecutionEnvironment = 'gpu', ...
    MiniBatchSize=32, ...
    MaxEpochs=10);
next2 = trainnet(augimdsTrain, next2,"crossentropy",options);

%%

load trainednet3.mat
load trainednet3_1.mat
%%

classNames = categories(imds.Labels);

YTest1 = minibatchpredict(next_net,augimdsTest);
YTest1 = scores2label(YTest1,classNames);
YTest2 = minibatchpredict(next2,augimdsTest);
YTest2 = scores2label(YTest2,classNames);

TTest = imdsTest.Labels;
figure
subplot(1, 2, 1);
confusionchart(TTest,YTest1);
title("next_net");
subplot(1, 2, 2);
confusionchart(TTest,YTest2);
title('next2');
%%

% im = imread("C:\Users\drake\MATLAB\Projects\PunchNet\PunchData\PunchData\Miss\Screenshot 2025-03-11 222732.png");
% 
% im = imresize(im,[224 224]);
% X = single(im);
% 
% scores1 = predict(trainednet3,X);
% trainednet4scores2 = predict(trainednet3_1,X);
% [label1,score1] = scores2label(scores1,classNames);
% [label2,score2] = scores2label(scores2,classNames);
% 
% figure
% imshow(im)
% title(['1: ' string(label1) + " (Score: " + gather(score1) + ")" ...
%     '2: ' string(label2) + " (Score: " + gather(score2) + ")"])

