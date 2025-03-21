%testing chunking logic
%synthetic sequence of punch predictions vv
%Frances Robertson

function Chunk(predictions) 

disp("Visualizing Predictions...")
YPred = predictions';

%convert to binary labels
isPunch = (YPred ~= "None");

frames = 1:numel(YPred);

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
hold on;

%different style
stem(frames, isPunch, 'b', 'LineWidth', 2, 'Marker', 'o', 'MarkerFaceColor','r', 'MarkerSize', 8);

%customization
xlabel('Frame Number', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('Punch Detected (1=yes, 0=no)', 'FontSize', 14, 'FontWeight', 'bold');
title('Punch Detection Over Time', 'FontSize', 16, 'FontWeight','bold');
grid on;
ylim([-0.2,1.2]); %keeps plot clean

%annotate punches
for i = 1:numel(YPred)
    if isPunch(i)
        text(frames(i), 1.05, YPred(i), 'FontSize', 12, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', 'Rotation', 45);
    end
end

hold off;

end