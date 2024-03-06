function [res]= clustermatch(indx, Y)
stream = RandStream.getGlobalStream;    %全局随机流
reset(stream);                  %重置随机流

[newIndx] = bestMap(Y,indx);
ACC = mean(Y==newIndx);
NMI = MutualInfo(Y,newIndx);
purity = purFuc(Y,newIndx);
[ARI,~,~,~] = valid_RandIndex(Y', newIndx');

res = [ACC, NMI, purity,ARI];