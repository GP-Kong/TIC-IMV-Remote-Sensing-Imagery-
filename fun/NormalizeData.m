function [ProcessData]=NormalizeData(X)

% % ProcessData = rescale(X, 0, 1, 'InputMin', min(X), 'InputMax', max(X));
% ProcessData = rescale(X, 0, 1);
[nFea,nSmp] = size(X);
for i = 1:nSmp
    ProcessData(:,i) = X(:,i) ./ max(1e-12,norm(X(:,i)));
end