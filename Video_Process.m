% Loads Frames into Datastore
folderName = "frame_Video_data_test_clip_MT";
input = imageDatastore(folderName);

%%
% load trainednet4.mat
ClassNames = ["Block", "Hit", "Miss", "None"];
ClassNames = categorical(ClassNames);

%%
% load next6.mat
YTest1 = minibatchpredict(next7_1,input);
%YTest1 = scores2label(YTest1);
% predictions = max(YTest1, [], 2)
predictions = scores2label(YTest1, ClassNames);

%%
framenum = 1:numel(input.Files);
blockframes = predictions == "Block";
hitframes = predictions == "Hit";
missframes = predictions == "Miss";
otherframes = predictions == "None";

%%
n = 1;

if numel(framenum(hitframes)) ~= 0
    figure(n)
    H = imtile(input.Files(framenum(hitframes)));
    title("hits")
    imshow(H);
    n = n + 1;
end

if numel(framenum(missframes)) ~= 0
    figure(n)
    M = imtile(input.Files(framenum(missframes)));
    title("miss")
    imshow(M);
    n = n + 1;
end

if numel(framenum(blockframes)) ~= 0
    figure(n)
    B = imtile(input.Files(framenum(blockframes)));
    title("block")
    imshow(B);
    n = n + 1;
end

% if numel(framenum(otherframes)) ~= 0
%     subplot(2, 2, n)
%     N = imtile(input.Files(framenum(otherframes)));
%     title("other");
%     imshow(N);
%     n = n + 1;
% end


% bar(framenum, predictions)
