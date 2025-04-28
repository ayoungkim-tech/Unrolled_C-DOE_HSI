function Ic = fftconv(I,k,shape)

fftsize = size(I)+size(k)-1;
FI = fft2(I,fftsize(1),fftsize(2));
FK = fft2(k,fftsize(1),fftsize(2));

FIc = bsxfun(@times, FI, FK);
Ic = real(ifft2(FIc));

if nargin >2 % Shape
    switch shape
        case 'full'
            Ic = Ic;
        case 'same'
            cut = ceil((size(k)-1)/2);
            Ic = Ic(cut(1)+(1:size(I,1)),cut(2)+(1:size(I,2)),:,:);
        case 'valid'
            cut = (size(k)-1);
            Ic = Ic(cut(1)+(1:size(I,1)-cut(1)),cut(2)+(1:size(I,2)-cut(2)),:,:);
        otherwise
            error('SHAPE must be "full", "same", or "valid".');
    end
else % Default is 'same'
    cut = ceil((size(k)-1)/2);
	Ic = Ic(cut+(1:size(I,1)),cut+(1:size(I,2)),:,:);
end

end