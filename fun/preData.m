function [splabels] = preData(data3D,num_Pixel,V)
if numel(data3D) < V
    error('preData:ViewCountMismatch', ...
        'data3D contains %d views, but V=%d.', numel(data3D), V);
end
if numel(num_Pixel) < V
    error('preData:AnchorCountMismatch', ...
        'num_Pixel must contain one value per view (expected %d, got %d).', ...
        V, numel(num_Pixel));
end
num_Pixel = num_Pixel(:)';

%% HSI data preprocessing
% Input:
%       data3D: 3D cube, HSI data.
%       num_Pixel: the number of superpixel.
% Output:
%       X: new data, 2D matrix. each column is a pixel 
%%
for v  = 1: V
    data = data3D{v};
    [nRow,nCol,dim] = size(data);
    X = reshape(data,nRow*nCol,dim);
    [X,~] = mapminmax(X);
    p = 1;
    start_pca = tic;
    coeff = pca(X);
    fprintf("pca,time:%f\n",toc(start_pca));
    Y_pca = X*coeff(:,1:p);
    img = im2uint8(mat2gray(reshape(Y_pca, nRow, nCol, p)));
    % ERS super-pixel segmentation.
    labels = mex_ers(double(img),num_Pixel(v));
    labels = labels + 1; 
    splabels{v} = labels;
end




 


end




