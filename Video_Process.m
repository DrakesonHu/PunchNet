function predictions = Video_Process(foldername)
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
disp("Making Predictions...")
YTest1 = minibatchpredict(next5,input);
predictions = scores2label(YTest1, ClassNames);
disp("Success!")

%% Assigns Predicted Hits to an Array

framenum = 1:numel(input.Files);
hitframes = predictions == "Hit";

%% Displays Predicted Frames

if numel(framenum(hitframes)) ~= 0
    figure
    H = imtile(input.Files(framenum(hitframes)));
    imshow(H);
    title("Hits")
end

end