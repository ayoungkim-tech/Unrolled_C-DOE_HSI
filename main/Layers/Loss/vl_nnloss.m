function Y = vl_nnloss(X,c,dzdy,varargin)

opts.loss = 'l2';
opts.regularizer = 'none';
opts.p = 0.8;
opts.r = 15;
opts.alpha = 0.001;
opts.crop = [0 0];
opts.vggnet = [];
opts.weights = {};
opts.vgglayers = {};
opts = vl_argparse(opts, varargin, 'nonrecursive') ;

% --------------------------------------------------------------------
cx = opts.crop(1);
cy = opts.crop(2);
X = X(cx+1:end-cx,cy+1:end-cy,:,:);
c = c(cx+1:end-cx,cy+1:end-cy,:,:);
if nargin <= 2 || isempty(dzdy)
    % Forward
    switch opts.loss
        case 'l2'
            t = ((X-c).^2)/2;
            Y = mean(t(:)); % reconstruction error per sample;
        case 'l1'
            Y = mean(abs(X(:)-c(:))); %sum(abs(X(:)-c(:))); % L1
        case 'ssim'
            Y = vl_ssim(X,c); %SSIM
        case {'vgg','VGG'}
            Y = vgg_loss_v2(X, c, opts.vggnet, opts.vgglayers);
        case {'vgg_lin','VGG_lin'}
            Y = vgg_loss_lin(X, c, opts.vggnet, opts.weights, opts.vgglayers);
        case 'rmse'
            t = ((X-c).^2);
            Y = sqrt(mean(t(:)));
        otherwise
            
    end
    % Regularizer
    switch opts.regularizer
        case {'DCT','dct'}
            r = dctsparsity(X);
            Y = Y+opts.alpha*r;
        case 'gradient'
            for ch = 1:size(X,3)
                for n = 1:size(X,4)
                    r(:,:,ch,n) = sparse_gradient(X(:,:,ch,n), [], opts.p, opts.alpha, c(:,:,ch,n));
                end
            end
            Y = Y+mean(r(:));
        case {'DarkChannel', 'darkchannel'}
            for n = 1:size(X,4)
                r(:,:,:,n) = DC_prior(X(:,:,:,n),[],opts.r,opts.alpha,c(:,:,:,n));
            end
            Y = Y+mean(r(:));
        case {'vgg','VGG'}
            r = vgg_loss_v2(X, c, opts.vggnet, opts.vgglayers);
            Y = Y + opts.alpha * r;
        case {'vgg_lin','VGG_lin'}
            r = vgg_loss_lin(X, c, opts.vggnet, opts.weights, opts.vgglayers);
            Y = Y + opts.alpha * r;
        otherwise
            
    end
    
else
    % Backward
    switch opts.loss
        case 'l2'
            Y = bsxfun(@minus,X,c).*dzdy./numel(X);
        case 'l1'
            Y = bsxfun(@times, sign(bsxfun(@minus,X,c)), dzdy);
        case 'ssim'
            Y = vl_ssim(X,c,dzdy); %SSIM
        case {'vgg','VGG'}
            Y = vgg_loss_v2(X, c, opts.vggnet, opts.vgglayers, dzdy);
        case {'vgg_lin','VGG_lin'}
            Y = vgg_loss_lin(X, c, opts.vggnet, opts.weights, opts.vgglayers, dzdy);
        case 'rmse'
            t = ((X-c).^2);
            dzdt = dzdy./numel(X).*(sqrt(mean(t(:)))+eps).^-1;
            Y = bsxfun(@minus,X,c).*dzdt;
        otherwise
            
    end
    % Regularizer
    switch opts.regularizer
        case {'DCT','dct'}
            r = dctsparsity(X,dzdy);
            Y = Y+opts.alpha*r;
        case 'gradient'
            for ch = 1:size(X,3)
                for n = 1:size(X,4)
                    dzdych = dzdy;%./(size(X,3)*size(X,4));
                    r(:,:,ch,n) = sparse_gradient(X(:,:,ch,n), dzdych, opts.p, opts.alpha, c(:,:,ch,n));
                end
            end
            Y = Y+r;
        case {'DarkChannel', 'darkchannel'}
            for n = 1:size(X,4)
                dzdyn = dzdy;%./size(X,4);
                r(:,:,:,n) = DC_prior(X(:,:,:,n),dzdyn,opts.r,opts.alpha,c(:,:,:,n));
            end
            Y = Y+r;
        case {'vgg','VGG'}
            r = vgg_loss_v2(X, c, opts.vggnet, opts.vgglayers, dzdy);
            Y = Y + opts.alpha * r;
        case {'vgg_lin','VGG_lin'}
            r = vgg_loss_lin(X, c, opts.vggnet, opts.weights, opts.vgglayers, dzdy);
            Y = Y + opts.alpha * r;
        otherwise
            
    end
    Y = padarray(Y,[cx cy],0,'both');
end

end


function y = sparse_gradient(x, dzdy, p, alpha, l)

backMode = ~isempty(dzdy);

[Gx_x,Gy_x] = imgradientxy(x,'intermediate');
[Gx_l,Gy_l] = imgradientxy(l,'intermediate');
Dx_l = exp(-10*(abs(Gx_l).^p));
Dy_l = exp(-10*(abs(Gy_l).^p));

if backMode
    % Derivative
    wx = Dx_l(:,1:end-1);
    wxplus = padarray(wx,[0 1],0,'pre');
    wxminus = padarray(wx,[0 1],0,'post');
    
    wy = Dy_l(1:end-1,:);
    wyplus = padarray(wy,[1 0],0,'pre');
    wyminus = padarray(wy,[1 0],0,'post');
    
    Gpx = (abs(Gx_x)+eps).^(p-1);
    Gpx = Gpx(:,1:end-1);
    Gpxplus = padarray(Gpx,[0 1],0,'pre');
    Gpxminus = padarray(Gpx,[0 1],0,'post');
    
    Gsx = sign(Gx_x);
    Gsx = Gsx(:,1:end-1);
    Gsxplus = padarray(Gsx,[0 1],0,'pre');
    Gsxminus = padarray(Gsx,[0 1],0,'post');
    
    Gpy = (abs(Gy_x)+eps).^(p-1);
    Gpy = Gpy(1:end-1,:);
    Gpyplus = padarray(Gpy,[1 0],0,'pre');
    Gpyminus = padarray(Gpy,[1 0],0,'post');
    
    Gsy = sign(Gy_x);
    Gsy = Gsy(1:end-1,:);
    Gsyplus = padarray(Gsy,[1 0],0,'pre');
    Gsyminus = padarray(Gsy,[1 0],0,'post');
    
    y = dzdy * alpha * p * (wxplus.*Gpxplus.*Gsxplus + wyplus.*Gpyplus.*Gsyplus...
        - wxminus.*Gpxminus.*Gsxminus - wyminus.*Gpyminus.*Gsyminus); %./numel(x);
else
    Dx_x = abs(Gx_x).^p;
    Dy_x = abs(Gy_x).^p;
    y = mean(alpha * (Dx_l(:)'*Dx_x(:) + Dy_l(:)'*Dy_x(:)));
end
end

function y = DC_prior(x, dzdy, r, alpha, l)

backMode = ~isempty(dzdy);

jx = dark_channel(x,r);
jl = dark_channel(l,r);
w = alpha*exp(-10.*jl);

if backMode
    dzdjx = dzdy*w.*sign(jx);%./numel(x);
    [~,y] = dark_channel(x,r,dzdjx);
else
    y = w.*abs(jx);
    y = mean(y(:)); %w(:)'*abs(jx(:));
end

end



function y = dctsparsity(x,dzdy)
y = zeros(size(x),'single');
if nargin <= 2 || isempty(dzdy)
    for ch = 1:size(x,3)
        for n = 1:size(x,4)
            y(:,:,ch,n) = dct2(x(:,:,ch,n));
        end
    end
    y = mean(abs(y(:)));
else
    for ch = 1:size(x,3)
        for n = 1:size(x,4)
            y(:,:,ch,n) = idct2(sign(dct2(x(:,:,ch,n))));
        end
    end
    y = bsxfun(@times, y, dzdy);%./numel(x);
end
end