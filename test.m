
clear;
close all;
clc
sigma = 0; % noise level
parallax = 0.25; 
fov = 90;
dir = 0; % gravity noise
focal = 1000;
K = [focal 0 0; 0 focal 0; 0 0 1];
num = 20; % number of points
iter = 1;
% sigma_pool = [0.5 1 1.5 2];
% parallax_pool = [0.25 0.6 1 2];
% fov_pool = [60 80 110 150];
% num_pool = [15 30 50 80];
% dir_pool = [0.01 0.05 0.1 0.2];

% sigma = sigma_pool(N);
% fov = fov_pool(N);
% num = num_pool(N);
% parallax = parallax_pool(N);
% dir = dir_pool(N);

[X1, X2, tr, X0, Tx1, Tz1, Tx2, Tz2] = generate_pt(sigma, parallax, focal,fov);

for kk = 1:iter   %
    ang1 = normrnd(0,dir); ang2 = normrnd(0,dir); ang3 = normrnd(0,dir); ang4 = normrnd(0,dir);

    Rg = tr(1:3,1:3,kk); % truth
    Tg = tr(1:3,4,kk);

    xc = X1(kk).data;
    xd = X2(kk).data;
    xc(3,:) = 1; xd(3,:) = 1;

    x1 = K\xc; x2 = K\xd;
    Ra = (Tz1(:,:,kk)*Tx1(:,:,kk)).';
    Rb = (Tz2(:,:,kk)*Tx2(:,:,kk)).';

    Ran = (Tz1(:,:,kk)*rotz(ang1)*rotx(ang2)*Tx1(:,:,kk)).';
    Rbn = (Tz2(:,:,kk)*rotz(ang3)*rotx(ang4)*Tx2(:,:,kk)).';

    x1n = Ran*x1;
    x2n = Rbn*x2;
    x1n = x1n./repmat(x1n(3,:),3,1);
    x2n = x2n./repmat(x2n(3,:),3,1);

    x3 = x1n; x4 = x2n; % normalized

    ind = randsample(size(x1,2),num);

    res = optimal_gravity_opt(x1n(1:2,ind), x2n(1:2,ind));
    rot_est = [1-res(1)^2 0 2*res(1); 0 1+res(1)^2 0; -2*res(1) 0 1-res(1)^2]/(1+res(1)^2);
    rot_error = norm(rot_est-Rg,'fro');
    trans_est = res(2:4);
    trans_error = acosd(abs(Tg'*trans_est)/(norm(Tg)*norm(trans_est)));
    fprintf('rotation error = %04f\n',rot_error);
    fprintf('translation error = %04f\n',trans_error);

    %% timing
    tic
    for i = 1:10000
        ind = randsample(size(x1,2),num);
        optimal_gravity_opt(x1n(1:2,ind), x2n(1:2,ind));
    end
    tt1 = toc;
    ave_time = tt1/10000*10^6;
    fprintf('average timings = %04f(us)\n',ave_time);


end