function [SimMat] = V1_LocalKernelCalculation(X, cla_num, rate)
num_views = length(X);
for v = 1 : num_views
    a = max(X{v}(:));
    X{v} = double(X{v}./a);
end
anchor_rate = rate;
opt1.style = 3;
opt1.IterMax = 50;
opt1.toy = 0;
[BViews] = FastmultiCLR(X, cla_num, anchor_rate, opt1, 10);
smp_num = size(X{1},1);
SimMat = zeros(smp_num, smp_num, num_views);
for v = 1 : num_views
    B = BViews{v};
    P = B * diag(sum(B,1))^-0.5;
    Current_kernel = P * P';
    SimMat(:,:,v) = Current_kernel;
end
end
