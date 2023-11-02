function [im1, im2, T, x0, Tx1, Tz1, Tx2, Tz2] = generate_pt(sigma, parallax,focal,fov)

n=200;

x0 = [rand(1,n)*6-3; rand(1,n)*6-3; rand(1,n)*4+2];

K = [focal 0 0; 0 focal 0; 0 0 1];
mm = 1000;
T = zeros(3,4,mm);
Tx1 = zeros(3,3,mm); Tx2 = zeros(3,3,mm);
Tz1 = zeros(3,3,mm); Tz2 = zeros(3,3,mm);
for i = 1:mm
    ay = rand*10-5; 
    ax1 = rand*20-10; az1 = rand*20-10; ax2 = rand*20-10; az2 = rand*20-10;

    T(1:3,1:3,i) = roty(ay);
    Tx1(:,:,i) = rotx(ax1);
    Tz1(:,:,i) = rotz(az1);
    Tx2(:,:,i) = rotx(ax2);
    Tz2(:,:,i) = rotz(az2);
    A = rand(3,1);
    T(1:3,4,i) = A/norm(A)*parallax; %
    
end

ff = focal*tand(fov/2);
for i = 1:mm
    
    x1a  = Tz1(:,:,i)*Tx1(:,:,i)*x0;
    x2a = Tz2(:,:,i)*Tx2(:,:,i)*(T(:,1:3,i)*x0+T(:,4,i));
    
    x1 = x1a./repmat(x1a(3,:),3,1);
    x2 = x2a./repmat(x2a(3,:),3,1);
    
    x1_image(:,:,i) = K*x1; % image points
    x2_image(:,:,i) = K*x2;
    
    x1_image_noi(1:2,:,i) = x1_image(1:2,:,i) + normrnd(0,sigma,2,n);%  
    x2_image_noi(1:2,:,i) = x2_image(1:2,:,i) + normrnd(0,sigma,2,n);%
    
    % feild of view  
    I = find(x1_image_noi(1,:,i)<ff & x1_image_noi(2,:,i)<ff & x1_image_noi(1,:,i)>-ff & x1_image_noi(2,:,i)>-ff ...
        & x2_image_noi(1,:,i)<ff & x2_image_noi(2,:,i)<ff & x2_image_noi(1,:,i)>-ff & x2_image_noi(2,:,i)>-ff);
    
    im1(i).data = x1_image_noi(:,I,i);
    im2(i).data = x2_image_noi(:,I,i);
    
end


end