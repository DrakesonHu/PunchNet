%testing chunking logic
%synthetic sequence of punch predictions vv
YPred = ["None", "None", "Hit", "Hit", "Hit", "None", "Hit", "Hit", "None", "Miss", "Miss", "Miss", "None","Block", "Block","None"];

%convert to binary labels
isPunch = (YPred ~= "None");

%display
disp("Frame-by-frame Predictions:");
disp(YPred);
disp("Binary Labels:");
disp(isPunch);

%identify punchstart and end frame
diffSeq = diff([0 isPunch 0]);
punchStart = find(diffSeq == 1);
punchEnd = find(diffSeq == -1) - 1;

%count number of punch chunks
numPunches = numel(punchStart);
fprintf("Total punches detected: %d\n", numPunches);

%display
disp ("Punch Start Frames:");
disp(punchStart);
disp("Punch End Frames:");
disp(punchEnd);

%plot timeline of punches
figure;
stem(1:numel(isPunch), isPunch, "filled");
xlabel('Frame Number');
ylabel('Punch Detected (1=yes, 0=no)');
title('Punch Detection');
grid on;