function batch = getBatch_edof(imdb, batch, varargin)

opts.useGpu = true;
opts = vl_argparse(opts, varargin);

N = numel(batch);

images = imdb.images.data(:,:,:,batch);

% rotate image randomly for data augmentation
r = randi([0 3]);
images = rot90(images, r);

% Assign random depth within the diopter range
diop = imdb.opts.diop;
if numel(diop)>2 % discrete
    depthbatch = randi(numel(diop),[1 N]);
    zinv = reshape(diop(depthbatch),1,1,1,N);
else % continuous
    zinv = (diop(1)-diop(2)) * rand(1,1,1,N,'like',images) + diop(2);
end
z = 1./zinv;
l = images;

if isfield(imdb.opts,'z_Ref')
    z_Ref = repmat(imdb.opts.z_Ref,1,1,1,N);
else
    z_Ref = 1./(0*zinv); % Reconstruction depth is infinity
end

if opts.useGpu
    images = gpuArray(images) ;
    z = gpuArray(z) ;
    l = gpuArray(l) ;
    z_Ref = gpuArray(z_Ref) ;
end

batch = { 'input', images, 'depth', z, 'depthRec', z_Ref, 'label', l };