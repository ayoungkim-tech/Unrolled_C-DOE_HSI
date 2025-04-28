clear all;
clc

% Some trick
gpus = [1];
fprintf('resetting GPU\n')
clear vl_tmove vl_imreadjpeg ;
gpuDevice(gpus)

%%
addpath(genpath('Layers'));
addpath(genpath('Util'));
addpath(genpath('..\HSI2RGB-master'));
graphpath = '..\graphviz-2.38\release\bin';
addpath(genpath('..\toolbox-master'));
addpath(graphpath);

%%
if ~(exist('imdb','var'))  
    in = load('TrainData\MS_256_31_ch');
    in = in.inputs;
    order  = randperm(size(in,4));
    in = in(:,:,:,order);
    imgsize = size(in);
    imdb = struct();
    imdb.images.set = ones(1, size(in, 4), 'single');
    valper = 10/100; % percentage of data to be used in validation
    val = randperm(size(in, 4),floor(size(in, 4)*valper));
    imdb.images.set(val) = 2;
    imdb.images.data = single(in);
    clear in;
end

diop = [2 2]; % Target depth range
z_f0 = 1/mean(diop); % Camera focus distance
imdb.opts.diop = diop;
imdb.opts.z_Ref = z_f0;

%% Some parameters

% Name of the network input and label images
% (Should be the same as inside getBatch function)
in = 'input';
label = 'label';
depth = 'depth';
% Heightmap noise std (min max)
h_std = [0 0]; %[20e-9 40e-9];
% Final image noise (should be zero I guess?)
n_std = [0 0];
% Loss
loss = 'l1'; 
% Regularization (loss) parameter alpha
alpha = 0.005;
% Radius (half window size) for the dark channel regularizer
r_dc = 17;
% Network options
batchSize = 1;
learningRate = 0.001;
weightDecay = 0.0001;
momentum = 0.5;
numEpochs = 16;
solver = @solver.adam;

% Color filter size
colfiltersize = [6 6];
% Color filter pixel size
X_x = 3.45e-6;
% DOE (Underlying) Sampling
p_xx = 9e-6;
% DOE Parameter space
p_xx_doe = 18e-6;

%% Network path

c = clock;
netpath = 'Net\NewFolder';
netpath = regexprep(netpath, '\s', '0');

net_dir = fullfile(pwd, netpath);
if ~exist(net_dir, 'dir')
    mkdir(net_dir);
end

%% Initialize network

net = dagnn.DagNN();
net.conserveMemory = false;

%% Camera 

% Camera struct
cam = Camera_Design(colfiltersize,p_xx,p_xx_doe,X_x,'z_f0',z_f0); %Change this

% PSF Layer
psflayer = psf_phi0_LRDOE('opts', cam, 'h_std', h_std); % Check this
net.addLayer('psflayer', psflayer, depth, 'psfs', 'phi0');

% PSF Layer for reconstruction network
psflayer = psf_phi0_LRDOE('opts', cam, 'h_std', h_std); % Check this
net.addLayer('psfInf', psflayer, 'depthRec', 'psfRec', 'phi0');

% Color Filter Layer
colorfilt = Color_Filter_Gauss_norm_activation('opts',cam,'activation','tanh');
net.addLayer('colfilt', colorfilt, {}, 'cfilter', {'sensor_mu','sensor_sigma','sensor_alpha'});

% Sensor Layer
sensorlayer = Phi('opts', cam, 's_std', n_std);
net.addLayer('sensor', sensorlayer, {in, 'psfs', 'cfilter'}, 'Isensor0');

% Clip
net.addLayer('SensorClip', Clip(), 'Isensor0', 'g');

output = 'g';

%% Main Network

%% Tentative reconstruction

% Phi Transpose
nch = cam.numchannel;
phiTlayer = PhiT('sc',nch);
net.addLayer('PhiT0', phiTlayer, {'g', 'psfRec', 'cfilter'}, 'f0');

%% Reconstruction

% ISTA net
L = 8; % number of iterations
C = 64; % Number of convolution filters in Prior
nch = cam.numchannel; % number of channels
initeps = 0.1; % Initial values for epsilon and eta
initeta = 0.5; 
initrho = 0.01;

for ll=1:L
      
    % Names
    fk = sprintf('f%d',ll-1); % Input    
    fk1 = sprintf('f%d',ll); % Output
    hk = sprintf('h%d',ll); % Prior
    hsk = sprintf('hs%d',ll); % Prior, scaled
    r0k = sprintf('r0%d',ll); % Residual
    rsk = sprintf('rs%d',ll); % Residual, scaled
    rk = sprintf('r%d',ll); % rk, Updated x with residual 
    ghat = sprintf('ghat%d',ll); % To calculate residual
    grk = sprintf('gr%d',ll); % To calculate residual
    
    % First, calculate the residual
    philayer = Phi('opts', cam, 's_std', [0 0]);
    net.addLayer(sprintf('Phi%d',ll), philayer, {fk, 'psfRec', 'cfilter'}, ghat);
    % Subtract
    net.addLayer(sprintf('Minus%d',ll), Subtract(), {output, ghat}, grk);

    phiTlayer = PhiT('sc',nch);
    net.addLayer(sprintf('PhiT%d',ll), phiTlayer, {grk, 'psfRec', 'cfilter'}, r0k);
    
    % From xk to rk in ISTA paper
    net.addLayer(sprintf('Scale_r_%d',ll),myScale('hasBias',false,'initsc',initrho), r0k, rsk, sprintf('rho_%d',ll));
    net.addLayer(sprintf('Sum_fr_%d',ll),dagnn.Sum(),{fk,rsk},rk);
    
    net = Spatial_Spectral_Prior_MS(net,nch,C,ll,rk,hk);
    
    % Addition
    net.addLayer(sprintf('Scale_h_%d',ll),myScale('hasBias',false,'initsc',initrho), hk, hsk, sprintf('rhoh_%d',ll));    
    net.addLayer(sprintf('ResSum_%d',ll),dagnn.Sum(),{rk,hsk},{fk1},{});
end

output = fk1;

%% Loss Layer

opts = {'regularizer', 'gradient3', 'alpha', alpha};
losslayer = dagnn.Loss('loss', loss, 'opts', opts);
net.addLayer('lossl1', losslayer, {output label}, 'objective');

%% Plot network and wait for input

wd = pwd;
cd(graphpath);
net.print('Format','dot','FigurePath',[wd '\' netpath '\net.pdf']);
cd(wd);

%%
net.initParams();

v = [1:9 numel(net.vars) + (-2:0)];
for vv = v
    net.vars(vv).precious = 1;
end

%%
opts = struct();
opts.expDir = netpath;
opts.batchSize = batchSize;
opts.numSubBatches = 1;
opts.learningRate = learningRate;
opts.weightDecay = weightDecay;
opts.momentum = momentum;
opts.numEpochs = numEpochs;
opts.solver = solver;
opts.gpus = gpus;
opts.train = [];
opts.plotStatistics = true;

cnn_train_dag_v3(net, imdb, @getBatch_edof, opts);

%% Save statistics of the epochs

figure(1); saveas(gcf,[netpath '\objective.png']);
clf;