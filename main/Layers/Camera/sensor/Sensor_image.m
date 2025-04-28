function [Ib,dedI,dedc] = Sensor_image(I,c,dedIb)
% Final monochromatic sensor image from multispectral data


%% Forward

% Pad image to integer multiple to the sensor color pattern, if necessary
padx = size(c,1) - mod(size(I,1),size(c,1));
padx = mod(padx,size(c,1));
pady = size(c,2) - mod(size(I,2),size(c,2));
pady = mod(pady,size(c,2));
Ip = padarray(I,[padx,pady],0,'post');

% Full response matrix, in image size
bx = size(Ip,1)/size(c,1);
by = size(Ip,2)/size(c,2);
cfull = repmat(c,bx,by);

% Monochromatic image
Ib = sum(bsxfun(@times, Ip, cfull),3);
Ib = Ib(1:size(I,1),1:size(I,2),:,:);

%% Backward
if nargin > 2

dedIb_p = padarray(dedIb,[padx,pady],0,'post');
dedI = bsxfun(@times, dedIb_p, cfull);
dedI = dedI(1:size(dedIb,1),1:size(dedIb,2),:,:);

dedc = bsxfun(@times, dedIb_p, Ip);
dedc = sum(sum(sum(reshape(dedc,size(c,1),bx,size(c,2),by,size(c,3),[]),2),4),6);
dedc = squeeze(dedc);

end



end