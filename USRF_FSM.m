function [Y, H, obj_main, changed] = USRF_FSM(Laplacian,Degree,NorKernel,class_num, theta, beta, NITR )
% USRF_HOLM
%
% Input:
%  Laplacian: the first-order and high-order Laplacian matrices.
%  Degree: the first-order and high-order degree matrices.
%  NorKernel: the first-order and high-order normalize similarity matrices.
%  class_num: class numbers.
%  theta:balance parameter.
%  beta: trade-off parameter.
%  NITR: Maximum interation times.
%
% Output:
%  Y: labels.
%  obj_main: Value of objective functions in each interation.
%  changed: number of nodes that change their labels in each Y-step.


if nargin<7
    NITR=30;
end

%% Initialization
n = size(Laplacian{1},1);
ker_num = size(Laplacian{1},3);

% Initialize H, Y
H = rand(n,class_num);
Y = diag(diag(H*H').^(-0.5))*H;
for i=1:size(Y,1)
    [~,mix]=max(Y(i,:));
    Y(i,:)=0;
    Y(i,mix)=1;
end

% Initialize L, N, D, W
alpha = ones(ker_num,1)/ker_num;

LC1 = mycombFun(Laplacian{1},alpha);  
LC2 = mycombFun(Laplacian{2},alpha);
L_star = LC1 + LC2;            % 计算普通拉普拉斯矩阵的合成矩阵

D1 = mycombFun(Degree{1}, alpha);
D2 = mycombFun(Degree{2}, alpha);
D = D1 + D2;                 % 计算初始合成度矩阵
Dia = sqrt(D);


L_sum = LsumForAlpha(Laplacian);  % 计算二阶拉普拉斯矩阵的和(更新alpha用)
N = construct_N(Laplacian,theta);


% Initialize paramters
NITR2=30;
G = Dia * H;
Y_temp = Y; 
obj_main = zeros(NITR+1,1);
changed = zeros(NITR,10);
obj = zeros(NITR2,NITR);
if nargin < 6
    beta=0.01;
end

%% Optimization process

for iter = 1:NITR
    K = Dia * Y_temp*(Y_temp'*D*Y_temp+eps*eye(class_num))^-0.5;
    obj_main(iter)=trace(H'*(L_star)*H - 2*beta*G'*K) + theta*alpha'*N*alpha;
    if abs(obj_main(iter+1)-obj_main(iter))<1e-3
        break;
    end
    err=1;
    t=1;  
    % gamma-step
    alpha = update_gamma(L_sum,H,N);
    LC1 = mycombFun(Laplacian{1},alpha);
    LC2 = mycombFun(Laplacian{2},alpha);
    L_star = LC1 + LC2;

    S1 = mycombFun(NorKernel{1},alpha);
    S2 = mycombFun(NorKernel{2},alpha);
    NorSimilarity = S1 + S2;

    D1 = mycombFun(Degree{1}, alpha);
    D2 = mycombFun(Degree{2}, alpha);
    D = D1 + D2;
    Dia = sqrt(D);
    G = Dia * H;

    % H-step
    while err>1e-3
        Z=2*(NorSimilarity)*G+2*beta*K;
        [U,~,V]=svd(Z,'econ');
        G=U*V';
        clear U V;
        obj(t,iter)=trace(G'*(NorSimilarity)*G+2*beta*G'*K);
        if t>=2
            err=abs(obj(t-1)-obj(t));
        end
        t=t+1;
        if t>NITR2
            break;
        end
    end
    H = Dia^(-1)*G;
    
    % Y-step
    HQ = G;
    [~, g] = max(HQ,[],2);
    Y_temp = TransformL(g, class_num);
    nn = sum(Y_temp.*(Y_temp));
    s = sum(HQ.*(Y_temp));
    [fq] = max(HQ,[],2);
    [~,idxi] = sort(fq);
    for it = 1:10
        converged=true;
        for ii = 1:n
            i = idxi(ii);
            dd = 1;
            hi = HQ(i,:);
            gi = Y_temp(i,:);
            [~,id0] = max(gi);
            ss = (s+(1-gi).*(hi))./sqrt(nn+(1-gi)) - (s-gi.*(hi))./(sqrt(nn-gi)+eps);
            [~,id] = max(ss);
            if id~=id0
                converged=false;
                changed(iter,it)=changed(iter,it)+1;
                gi = zeros(1,class_num);
                gi(id) = 1;
                Y_temp(i,:) = gi;
                nn(id0) = nn(id0) - dd;  nn(id) = nn(id) + dd;
                s(id0) = s(id0) - HQ(i,id0);  s(id) = s(id) + HQ(i,id);
            end
        end
        if converged
            clear s nn;
            break;
        end
    end
end

Ypred_m = zeros(n,1);
for i = 1:(n)
    [~, Ypred_m(i)] = max(Y_temp(i,:));
end
Y=Ypred_m;
end

function L_sum = LsumForAlpha(Lapla)
order = size(Lapla,2);
m = size(Lapla{1},3);
d = size(Lapla{1},2);
n = size(Lapla{1},1);
L_sum = zeros(n,d,m);
for kk = 1:order
    L_temp = Lapla{kk};
    for tt = 1:m
        L_sum(:,:,tt) = L_sum(:,:,tt) + L_temp(:,:,tt);  % 
    end
end
end
