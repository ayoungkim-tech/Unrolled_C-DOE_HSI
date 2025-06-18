% This work is licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.
% See https://creativecommons.org/licenses/by-nc-nd/4.0/
%
% © 2025 IEEE. Personal use of this material is permitted.
% For any other use, permission must be obtained from IEEE.
%
% This code is supplementary material for the article:
% "Snapshot Hyperspectral Imaging with Co-designed Optics, Color Filter Array, and Unrolled Network", 
% by Ayoung Kim, Ugur Akpinar, Erdem Sahin, Atanas Gotchev, 
% published in IEEE Open Journal of Signal Processing, 2025.

clear all;
warning off;
clc;

%% Add necessary directories to path

addpath(genpath('Layers'));
addpath(genpath('HSI2RGB-master'));
addpath(genpath('Util'));
addpath(genpath('..\HSI2RGB-master'));
graphpath = '..\graphviz-2.38\release\bin';
addpath(genpath('..\toolbox-master'));

%% Read and load network

save_dir = 'Result';
epoch_dir = 'Net\';

load([epoch_dir 'net-epoch-16.mat']);

% Load network
net = dagnn.DagNN.loadobj(net);

net.mode = 'test';
net.conserveMemory = false;

net.vars(net.getVarIndex('g')).precious = 1;
net.vars(end-2).precious = 1;

% Camera parameters
opts = net.layers(1).block.opts;
nch = opts(1).numchannel;

%% Sharp multispectral image and depth

lambda_MS = [opts.lambda]*1e+9;
% Depth
diop = 2;
z = 1./diop;
z_ref = opts(12).z_f;

%% Evaluation through network: Sensor image formation and reconstruciton altogether

img_dir = 'TestData\ICVL\';
files = dir(fullfile(img_dir, '*.mat'));
file_names = {files.name};

n_test_img = numel(files);
img_sz = [512,512];

psnr_vals = zeros(1,n_test_img);
mvssim_vals = zeros(1,n_test_img);
sam_vals = zeros(1,n_test_img);
Iin_vals = zeros(n_test_img, img_sz(1), img_sz(2), nch);
Iin_rgb_vals = zeros(n_test_img, img_sz(1), img_sz(2), 3);
Iout_vals = zeros(n_test_img, img_sz(1), img_sz(2), nch);
Iout_rgb_vals = zeros(n_test_img, img_sz(1), img_sz(2), 3);

for idx = 1 : n_test_img
    % load test data
    filename = files(idx).name;
    load([img_dir,filename]);
    Iin = single(rad);
    Iin = imresize(Iin,img_sz);
    % normalize the Iin vals
    Iin = Iin ./max(Iin(:));

    % Main part: evaluate input
    tic;
    net.eval({'input', Iin, 'depth', z, 'depthRec', z_ref, 'label', Iin});
    toc

    % Results
    Ib = net.vars(net.getVarIndex('g')).value; % Sensor image (single channel, multiplexed)
    Iout = squeeze(net.vars(end-2).value); % Reconstructed MS output

    %% Input and output, rgb images

    % Input
    [ydim,xdim,zdim]=size(Iin);
    % reorder data so that each column holds the spectra of of one pixel
    Z = reshape(Iin,[],zdim);
    % use the D65 illuminant
    illuminant=65;
    % do minor thresholding
    threshold=0.001;
    %Create the RBG image,
    Iin_rgb = HSI2RGB(lambda_MS,Z,ydim,xdim,illuminant,threshold);

    % Output
    Z = reshape(Iout,[],zdim);
    Iout_rgb = HSI2RGB(lambda_MS,Z,ydim,xdim,illuminant,threshold);

    PSNR = psnr(Iin,Iout);
    MvSSIM = mvssim(Iin, Iout);
    SAM_MAP = sam(Iin, Iout);

    % save vals
    psnr_vals(idx) = PSNR;
    mvssim_vals(idx) = MvSSIM;
    sam_vals(idx) = mean(SAM_MAP(:));
    Iin_vals(idx,:,:,:) = Iin;
    Iin_rgb_vals(idx,:,:,:) = Iin_rgb;
    Iout_vals(idx,:,:,:) = Iout;
    Iout_rgb_vals(idx,:,:,:) = Iout_rgb;

    figure(1);
    imshow([Iin_rgb Iout_rgb])
    title(PSNR);
end

avg_psnr = round(mean(psnr_vals(:)),2)
avg_mvssim = round(mean(mvssim_vals(:)),2)
avg_sam = round(mean(sam_vals(:)),2)

%% Save result in mat files
save([save_dir,'\psnr_vals.mat'], 'psnr_vals');
save([save_dir,'\mvssim_vals.mat'], 'mvssim_vals');
save([save_dir,'\sam_vals.mat'], 'sam_vals');
save([save_dir,'\Iin_vals.mat'], 'Iin_vals');
save([save_dir,'\Iin_rgb_vals.mat'], 'Iin_rgb_vals');
save([save_dir,'\Iout_vals.mat'], 'Iout_vals');
save([save_dir,'\Iout_rgb_vals.mat'], 'Iout_rgb_vals');

%% Save other parameters

phi0 = net.getParam('phi0').value;
save([save_dir,'\phi0.mat'], 'phi0');
colfilt = net.vars(5).value;
[mu, sigma,~] = net2colfilt(net);
% colfilt rgb img
[ydim,xdim,zdim]=size(colfilt);
Z = reshape(colfilt,[],zdim);
colfilt_rgb = HSI2RGB(lambda_MS,Z,ydim,xdim,illuminant,threshold);
save([save_dir,'\colfilt.mat'], 'colfilt','colfilt_rgb','mu','sigma');
% save heightmap
cam = net.layers(1).block.opts;
save([save_dir,'\cam.mat'], 'cam');
U1x = cam(1).U1x;
U1y = cam(1).U1y;
phi0_hr = U1x*phi0*U1y';
ch0 = cam(1).ch0;
lambda = cam(ch0).lambda;
k = 2*pi./lambda;
n = cam(ch0).ref_idx;
heightmap = phi0_hr./k./(n-1);
heightmap = heightmap - min(heightmap(:));
heightmap = heightmap.*cam(1).M;
figure(3);imagesc(heightmap);colorbar;
save([save_dir,'\heightmap.mat'], 'heightmap');

%% loss curve

plot([stats.train.objective],'-o')
hold on;
plot([stats.val.objective],'-o')
hold off;
title('loss curve');