%% Generate Laplican matrix for each view
%% This part of the reference article
% Zhou, S., et al.: Multi-view spectral clustering with optimal neighborhood Laplacian matrix.
% In: Proceedings of the Thirty-Fourth AAAI Conference on Artificial Intelligence, pp. 6965–6972 (2020)

function [ho_Lapla, ho_Degree, ho_Kernel] = laplacian_generation(BseKer)

ker_num = size(BseKer, 3);
smp_num = size(BseKer, 1);
Lapla = zeros(smp_num, smp_num, ker_num);
Degree_mat = zeros(smp_num, smp_num, ker_num);
Nor_kernel = zeros(smp_num, smp_num, ker_num);

for i = 1 : ker_num
    cur_ker = BseKer(:,:,i);
    [laplacian, degree, norker] = LaplicanGeneration(cur_ker);
    Lapla(:,:,i) = laplacian;
    Degree_mat(:,:,i) = degree;
    Nor_kernel(:,:,i) = norker;
end

ho_Lapla = cell(1, 2);
ho_Degree = cell(1, 2);
ho_Kernel = cell(1, 2);
ho_Lapla{1} = Lapla;
ho_Degree{1} = Degree_mat;
ho_Kernel{1} = Nor_kernel;

for i = 1 : ker_num
    [laplacian, degree, norker] = HO_laplacian(Lapla(:,:,i));
    Lapla(:,:,i) = laplacian;
    Degree_mat(:,:,i) = degree;
    Nor_kernel(:,:,i) = norker;
end

ho_Lapla{2} = Lapla;
ho_Degree{2} = Degree_mat;
ho_Kernel{2} = Nor_kernel;
end


%% Generate Laplican matrix of K according to index matrix A
function [laplacian, degree, normalize_kernel] = LaplicanGeneration(K)
eye_matrix = 1 - eye(size(K));   % 对角线为0
K = K .* eye_matrix;
c_diag = sum(K, 2);
c_diag(c_diag == 0) = 1;
c_diag(c_diag < 10^(-10)) = 10^(-10);
% degree = diag(c_diag);
c_diag = diag(sqrt(c_diag.^(-1)));
degree = c_diag;
laplacian = degree - K;   % L = D-W
normalize_kernel = c_diag * K * c_diag;  % D^(-(1/2))*W*D^(-(1/2))
end

%% Generate high-order laplacian
function [L_record, degree, normalize_kernel] = HO_laplacian(L_record)
L_record = diag(diag(L_record)) - L_record;

% imagesc(L_record)
L_record = L_record * L_record;
L_record_P = L_record(L_record > 0);
L_record_org = L_record;
mean_value = mean(L_record_P);
std_value = std(L_record_P);
L_record(L_record < mean_value - std_value/2) = 0;
[L_record, degree, normalize_kernel] = LaplicanGeneration(L_record);

L_record = kernel_completion(L_record, L_record_org);
% figure()
% imagesc(L_record)
end

function kernel = kernel_completion(kernel, org_kernel)

Avg_index = kernel ~= 0;
Ker_sum = sum(Avg_index, 2);
index = find(Ker_sum == 0);
threshold = mean(kernel(:));
if ~isempty(index)
    org_kernel = org_kernel - diag(diag(org_kernel));
    Small_samples = org_kernel(index, :);
    [~, smp_indexes] = sort(Small_samples,2,'descend');
    smp_indexes = smp_indexes(:,1);
    
    for ii = 1 : size(Small_samples, 1)
        kernel(index(ii), smp_indexes(ii)) = threshold;
    end
end
end