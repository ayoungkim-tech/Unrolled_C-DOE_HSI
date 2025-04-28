function [In,de_dI] = normalize_ch_v2(I,method,de_dIn)

% Normalize channel-wise, so that the summation is 1
% Extention: 'method' defines if the summation is one for all pixels, or
% only for the maximum (i.e. sum <= 1)

% Normalization factor
normfactor = sum(I,3)+eps;

% Maximum
[maxn,dmaxndI] = max2d_idx(normfactor);

% Forward
switch lower(method)
    case 'sum'
        In = bsxfun(@times,I,1./normfactor);
    case 'max'
        In = bsxfun(@times,I,1./maxn);
    otherwise
        error('Unknown normalization. The options are: max or sum.');
end

% Backward
if nargout > 1
    switch lower(method)
        case 'sum'
            de_dI = bsxfun(@times, de_dIn, 1./normfactor) - bsxfun(@times, sum(de_dIn.*In,3), 1./normfactor);
        case 'max'
            de_dI1 = bsxfun(@times, de_dIn, 1./maxn); % from I/maxn, easy
            de_dI2 = dmaxndI .* bsxfun(@times, sum(sum(sum(de_dIn.*In))), 1./maxn); % derivative of 1/maxn;
            de_dI = de_dI1 - de_dI2;
    end
end

end

function [maxn,dmaxndI] = max2d_idx(II)
% Maximum with derivative

% Take the maximum, with the indices
[maxx,i] = max(II);
[maxn,j] = max(maxx);

% Assign 1 to the corresponding (max) indices, 0 to others
dmaxndI = zeros(size(II),'like',II);
for b = 1:size(II,4) % over batch
    % Indices
    jb = j(b); 
    ib = i(:,jb,:,b);
    dmaxndI(ib,jb,:,b) = 1;
end

end