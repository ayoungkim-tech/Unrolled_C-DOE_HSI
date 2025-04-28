function ref_idx = n_MA_P1200G(lambda_q)

% lambda - nanometers

data = importdata('MA_P1200G-dispersion-visible.txt');
data = data.data;
lambda = data(:,1);
n = data(:,2);
k = data(:,3);

n_q=interp1(lambda,n,lambda_q,'linear',0);
k_q=interp1(lambda,k,lambda_q,'linear',0);

ref_idx= n_q - 1i*k_q;
end