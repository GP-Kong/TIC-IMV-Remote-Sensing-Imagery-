function labels = make_feature_anchor_labels(dataFull, V, defaultAnchorNum, expectedN)
%MAKE_FEATURE_ANCHOR_LABELS Build fixed anchor labels without image geometry.
% Each returned vector contains labels in 1:m, where m is the requested
% anchor count. This is used for irregular labeled-pixel subsets whose
% imageSize is intentionally empty.
if ~iscell(dataFull) || numel(dataFull) < V
    error('dataFull must contain at least V feature views.');
end
if nargin < 4 || isempty(expectedN)
    expectedN = size(dataFull{1}, 2);
end
expectedN = double(expectedN);

labels = cell(1, V);
for v = 1:V
    X = double(dataFull{v});
    if size(X, 2) ~= expectedN && size(X, 1) == expectedN
        X = X';
    elseif size(X, 2) ~= expectedN
        error('View %d has %d samples; expected %d.', ...
            v, size(X, 2), expectedN);
    end
    N = expectedN;
    if isscalar(defaultAnchorNum)
        requestedM = double(defaultAnchorNum);
    else
        if numel(defaultAnchorNum) < v
            error('defaultAnchorNum must be scalar or contain one value per view.');
        end
        requestedM = double(defaultAnchorNum(v));
    end
    m = min(requestedM, N);
    if m < 2
        error('Each view must contain at least two samples for anchor labels.');
    end
    X(~isfinite(X)) = 0;
    X = NormalizeData(X);
    [label, ~] = litekmeans(X', m, 'replicates', 1, 'maxiter', 50);
    labels{v} = double(label(:));
end
end
