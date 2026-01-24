directory = uigetdir();
files = dir(directory);
tokens = [".", "..", "Thumbs.db"];
to_remove=[];
for i = 1:length(files)
    for token = tokens
        if matches(files(i).name, token)
            to_remove(end+1)=i;
        else
            continue
        end
    end
end
for rm = flip(to_remove)
    files(rm) = [];
end

%sz = [52 434];
%varTypes = ["string", "uint64"];
%data = table('Size',sz,'VariableTypes',varTypes);
data = table;

for i = 1:numel(files)
    filename = fullfile(directory, files(i).name);
    toadd = readtable(filename);
    
end