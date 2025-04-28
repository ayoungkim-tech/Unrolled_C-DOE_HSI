function SAM_map_deg = sam(H1, H2)
    % Compute SAM between two hyperspectral image cubes (H1 and H2).
    % H1, H2: Hyperspectral cubes of size (rows x cols x bands).
    % Output: SAM_map of size (rows x cols) with angle values in radians.
    
    % Ensure both cubes have the same dimensions
    if size(H1) ~= size(H2)
        error('H1 and H2 must have the same dimensions');
    end
    
    % Get dimensions
    [rows, cols, bands] = size(H1);
    
    % Reshape to (pixels x bands) for vectorized computation
    H1_reshaped = reshape(H1, [], bands);
    H2_reshaped = reshape(H2, [], bands);
    
    % Compute dot product for each pixel
    dot_products = sum(H1_reshaped .* H2_reshaped, 2);
    
    % Compute norms
    norms_H1 = sqrt(sum(H1_reshaped .^ 2, 2));
    norms_H2 = sqrt(sum(H2_reshaped .^ 2, 2));
    
    % Prevent division by zero
    valid_pixels = (norms_H1 > 0) & (norms_H2 > 0);
    
    % Compute SAM in radians
    SAM_values = NaN(size(dot_products)); % Initialize with NaN
    SAM_values(valid_pixels) = acos(dot_products(valid_pixels) ./ ...
        (norms_H1(valid_pixels) .* norms_H2(valid_pixels)));
    
    % Reshape back to original image size
    SAM_map = reshape(SAM_values, rows, cols);
    SAM_map_deg = rad2deg(SAM_map);
end