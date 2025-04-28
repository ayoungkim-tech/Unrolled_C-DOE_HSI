function n = n_silica(lambda)

% lambda[um]

%https://refractiveindex.info/?shelf=main&book=SiO2&page=Malitson
n=sqrt( 1 + 0.6961663./(1-(0.0684043./lambda).^2) + ...
            0.4079426./(1-(0.1162414./lambda).^2) + ...
            0.8974794./(1-(9.896161./lambda).^2) );

end

