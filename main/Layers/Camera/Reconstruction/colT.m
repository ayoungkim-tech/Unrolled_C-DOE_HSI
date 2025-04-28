function [I_HS,dedI,dedc] = colT(I,c,dedIHS)
% Transpose of the sensor response operator


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

% Reconstructed HS image
I_HS = bsxfun(@times, Ip, cfull);
I_HS = I_HS(1:size(I,1),1:size(I,2),:,:);

%% Backward
if nargin > 2

dedIHS_p = padarray(dedIHS,[padx,pady],0,'post');
dedI = sum(bsxfun(@times, dedIHS_p, cfull),3);
dedI = dedI(1:size(dedIHS,1),1:size(dedIHS,2),:,:);

dedc = bsxfun(@times, dedIHS_p, Ip);
dedc = sum(sum(sum(reshape(dedc,size(c,1),bx,size(c,1),by,size(c,3),[]),2),4),6);
dedc = squeeze(dedc);

end



end