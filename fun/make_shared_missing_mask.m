function [M, meta] = make_shared_missing_mask(N, V, setting, seed, mode)
%MAKE_SHARED_MISSING_MASK Generate one mask to be shared by all algorithms.
% mode='exact' uses the supplied per-view missing rates. mode='tic_imv' uses
% a TIC-IMV-style random alpha vector and setting as the overall rate r.
% M is N-by-V with 1=observed and 0=missing. Actual rates are returned.
if nargin < 5 || isempty(mode), mode='exact'; end
if nargin < 4 || isempty(seed), seed=1; end
rng(seed,'twister');
mode=lower(char(mode));
if strcmp(mode,'exact')
    target=double(setting(:)');
    if numel(target)~=V, error('Exact setting must contain V rates.'); end
    alpha=[];
elseif strcmp(mode,'tic_imv')
    r=double(setting);
    if ~isscalar(r) || r<0 || r>=1, error('TIC-IMV setting must be overall rate in [0,1).'); end
    alpha=0.25+rand(1,V);
    alpha=V*alpha/sum(alpha);
    target=min(0.98,r*alpha);
else
    error('mode must be exact or tic_imv.');
end
target=min(max(target,0),0.98);
observedCount=max(1,N-round(target*N));
if sum(observedCount)<N
    error('These rates cannot preserve all view marginals and keep one observed view per sample.');
end
% Assign one observed view to each sample while respecting the exact column
% quotas, then fill the remaining per-view quotas on random eligible rows.
primaryCount=floor(N*observedCount/sum(observedCount));
left=N-sum(primaryCount);
fraction=N*observedCount/sum(observedCount)-primaryCount;
while left>0
    eligible=find(primaryCount<observedCount);
    [~,ord]=sort(fraction(eligible),'descend');
    j=eligible(ord(1)); primaryCount(j)=primaryCount(j)+1;
    fraction(j)=-inf; left=left-1;
end
primary=zeros(N,1); rows=randperm(N); pos=1;
for v=1:V
    count=primaryCount(v);
    if count>0
        primary(rows(pos:pos+count-1))=v;
        pos=pos+count;
    end
end
M=false(N,V);
M(sub2ind([N,V],(1:N)',primary))=true;
for v=1:V
    residual=observedCount(v)-primaryCount(v);
    if residual>0
        eligible=find(~M(:,v));
        chosen=eligible(randperm(numel(eligible),residual));
        M(chosen,v)=true;
    end
end
actual=1-mean(M,1);
meta=struct('mode',mode,'seed',seed,'targetMissingRate',target, ...
    'actualMissingRate',actual,'observationRate',mean(M,1),'alpha',alpha, ...
    'averageMissingRate',1-mean(M(:)), ...
    'N',N,'V',V,'maskConvention','N-by-V; 1 observed, 0 missing');
end
