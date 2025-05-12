function SAM_map = sam(H1, H2)
    % Compute SAM between two hyperspectral image cubes (H1 and H2).
    % H1, H2: Hyperspectral cubes of size (rows x cols x bands).
    % Output: SAM_map of size (rows x cols) with angle values in degrees.
    
     % Get dimensions
    [rows, cols, bands] = size(H1);
    SAM_map = zeros(rows, cols);
    
    H1 = H1 + eps;
    H2 = H2 + eps;
    
    for c=1:cols
        for r=1:rows
            v1 = double(squeeze(H1(r,c,:)));
            v2 = double(squeeze(H2(r,c,:)));
    
            den = sqrt(sum(v1.^2)) .* sqrt(sum(v2.^2));
            num = dot(v1,v2);
            ang = acos((num)./(den));
            deg = rad2deg(ang);
    
            SAM_map(r,c) = real(deg);
        end
    end
end