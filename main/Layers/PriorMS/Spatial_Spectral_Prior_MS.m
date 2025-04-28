function net = Spatial_Spectral_Prior_MS(net,f0,f1,s,inname,outname)

% Prior Network from Wang et.al.: Hyperspectral Image Reconstruction Using a Deep Spatial-Spectral Prior
% f0: Number of input channels
% f1: Number of convolution filters
% s: Stage (iteration) number
% inname: Input name
% Outname: Output name

% First Convolution
convObj = dagnn.Conv('size', [3 3 f0 f1], 'pad', floor([3 3]./2), 'hasBias', true,'opts', {});
net.addLayer(sprintf('conv3x3_%d_1',s), convObj, {inname}, {sprintf('x_%d_1',s)}, {sprintf('f_%d_1',s), sprintf('b_%d_1',s)});
net.addLayer(sprintf('relu_%d_1',s), dagnn.ReLU(), {sprintf('x_%d_1',s)}, {sprintf('x_%d_2',s)}, {});

% Second Convolution
convObj = dagnn.Conv('size', [3 3 f1 f0],'pad', floor([3 3]./2), 'hasBias', true,'opts', {});
net.addLayer(sprintf('conv3x3_%d_2',s), convObj, {sprintf('x_%d_2',s)}, {sprintf('x_%d_3',s)}, {sprintf('f_%d_2',s), sprintf('b_%d_2',s)});

% Addition
net.addLayer(sprintf('SpaSum_%d',s),dagnn.Sum(),{inname,sprintf('x_%d_3',s)},sprintf('Spaout_%d',s),{});

% Spectral Convolution
convObj = dagnn.Conv('size', [1 1 f0 f0], 'hasBias', true,'opts', {});
net.addLayer(sprintf('conv3x3_%d_3',s), convObj, {sprintf('Spaout_%d',s)}, {outname}, {sprintf('f_%d_3',s), sprintf('b_%d_3',s)});

end