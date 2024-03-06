clear all
clc
warning off;
addpath(genpath('./ClusteringEvaluation'));
addpath(genpath('./SimilarityMatrixConstruction'));
path_data = './Datasets/';
res_path = './Result_temp/';

DataName = {'Yale'};


datacnt = size(DataName,2);
for name = 1:datacnt
    load([path_data, DataName{name}],'X','truth');
    numclass = length(unique(truth));
    n = size(X,1);
    ker_num = size(X,2);
    

    runcount = 1;
    NNrate = 0.1:0.1:0.9;
    theta = 2.^(-15:3:15);
    beta = 0.^(-4:1:0);
    DetailResult = cell(1,length(NNrate));
    result = zeros(length(theta),length(beta),4);

    for i = 1:length(NNrate)   
            [SimMat] = V1_LocalKernelCalculation(X, numclass, NNrate(i));
            [Laplacian, Degree, NorKernel] = laplacian_generation(SimMat);
            for para1 = 1:length(theta)
                for para2 = 1:length(beta)
                    fprintf(1, 'running the proposed algorithm with theta %d..., beta %d...\n', theta(para1),beta(para2));
                    [Ypre, H, obj_main, changed] = USRF_FSM(Laplacian,Degree,NorKernel,numclass,theta(para1),beta(para2));
                    result(para1,para2,:) = clustermatch(Ypre, truth)
                end
            end
            DetailResult{i} = result;
           
            max_res = max(max(result));
            res(1) = max_res(:,:,1);
            res(2) = max_res(:,:,2);
            res(3) = max_res(:,:,3);
            res(4) = max_res(:,:,4);
            res

            fid = fopen('0227_result.txt','a');
            fprintf(fid,'%s\t %8.4f %8.4f %8.4f %8.4f %8.4f \n', DataName{name},NNrate(i),res);  
            fclose(fid);      
    end   
    save([res_path,DataName{name},'_result.mat'],'DetailResult');
end