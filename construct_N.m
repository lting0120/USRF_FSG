function N = construct_N(H,lambda_value)
order = size(H,2);
H_matrix_N = cell(1,order);
N = zeros(size(H{1},3));
for o = 1:order
    H_matrix_N{o} = Cor_Calculation(H{o});
    N = N + H_matrix_N{o};
end
N = N*lambda_value;
end

function H_Correlation = Cor_Calculation(H)
ker = size(H,3);
for ii = 1:ker
    for jj = ii:ker
        a = norm(H(:,:,ii),'fro');
        b = norm(H(:,:,jj),'fro');
        H_Correlation(ii,jj) = trace(H(:,:,ii)'*H(:,:,jj))/(a*b);
    end
end
H_Correlation = (H_Correlation+H_Correlation')/2;
end