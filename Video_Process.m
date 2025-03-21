% function predictions = Video_Process(foldername)
%% This script is used to process input videos using a trained neural network
% written in March 2025
% Drakeson Hu

%% Loads Frames into Datastore

input = imageDatastore(foldername);

%% Creates Class Names to Assign Predictions to

ClassNames = ["Block", "Hit", "Miss", "None"];
ClassNames = categorical(ClassNames);

%% Makes Predictions

disp("Loading Neural Network...")
load next5.mat
disp("Success")
disp("Making Predictions...")
YTest1 = minibatchpredict(next5,input);
predictions = scores2label(YTest1, ClassNames);

%% Assigns Each Predicted Class to an Array

framenum = 1:numel(input.Files);
blockframes = predictions == "Block";
hitframes = predictions == "Hit";
missframes = predictions == "Miss";
% There is no need to create a category for the "None" class, as there is
% nothing happening during those frames

%% Displays Predicted Frames

n = 1;

if numel(framenum(hitframes)) ~= 0
    figure(n)
    H = imtile(input.Files(framenum(hitframes)));
    imshow(H);
    title("hits")
    n = n + 1;
end

if numel(framenum(missframes)) ~= 0
    figure(n)
    M = imtile(input.Files(framenum(missframes)));
    imshow(M);
    title("miss")
    n = n + 1;
end

if numel(framenum(blockframes)) ~= 0
    figure(n)
    B = imtile(input.Files(framenum(blockframes)));  
    imshow(B);
    title("block")
end

% end