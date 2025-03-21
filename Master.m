[file,filepath] = uigetfile({'*.avi;*.mp4;*.m4v;*.mpg;*.mov'}, ...
    'Video Files (*.avi, *.mp4, *.m4v, *.mpg, *.mov)');

file_location = [filepath, file];

%%

disp("creating datastore for analysis")
foldername = Video_reader(file_location, file);

%%
predictions = Video_Process(foldername);

%%
Chunk(predictions);