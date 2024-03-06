function gamma = update_gamma(L_sum,W,M)
m = size(L_sum,3);
f = zeros(m,1);

for i = 1:m
    f(i) = trace(W'*L_sum(:,:,i)*W);
end
options = optimset('Display','off');
gamma = quadprog(M/2,-f,[],[],ones(1,m),1,zeros(m,1),ones(m,1),[],options);
end
