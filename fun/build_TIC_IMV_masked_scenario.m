function outputFile = build_TIC_IMV_masked_scenario(sourceScenarioFile, overallRate, seed, outputFile)
%BUILD_TIC_IMV_MASKED_SCENARIO Apply one TIC-IMV-style random-alpha mask to a split.
% The source is a prepared scenario MAT file containing data_full and labels.
% Its full data are used only to reconstruct the same underlying dataset;
% every baseline and proposed method must then read outputFile unchanged.
if nargin<3 || isempty(seed), seed=1; end
source=load(sourceScenarioFile);
if ~isfield(source,'data_full')
    error('Source MAT must include data_full; pass a scenario from the preparation script.');
end
if isfield(source,'truelabel')
    labels=source.truelabel;
elseif isfield(source,'gt')
    labels=source.gt;
else
    error('Source MAT must include truelabel or gt.');
end
% Keep only fields needed by the solver and evaluation. Do not copy an old
% masked data cell or redundant data_full_NxD arrays into every repeat.
S=struct('data_full',{source.data_full},'truelabel',labels,'gt',labels);
keep={'dataset','scenario','viewNames','evalMask','imageSize','clusterNum'};
for k=1:numel(keep)
    if isfield(source,keep{k}), S.(keep{k})=source.(keep{k}); end
end
V=numel(S.data_full); N=size(S.data_full{1},2);
[M,meta]=make_shared_missing_mask(N,V,overallRate,seed,'tic_imv');
S.data=S.data_full;
for v=1:V, S.data{v}(:,~M(:,v)')=NaN; end
S.data_NxD=transpose_cells(S.data);
S.MissingStatus=uint8(M);
S.missingMask=uint8(~M);
S.targetMissingRate=meta.targetMissingRate;
S.actualMissingRate=meta.actualMissingRate;
S.averageMissingRate=meta.averageMissingRate;
S.observationRate=meta.observationRate;
S.index=cell(1,V);
for v=1:V, S.index{v}=find(M(:,v)); end
S.N=N; S.V=V; S.seed=seed; S.alpha=meta.alpha;
S.maskProtocol='TIC-IMV-style random alpha; one shared exact-marginal mask';
S.maskMeta=meta;
if nargin<4 || isempty(outputFile)
    root=fileparts(mfilename('fullpath')); repoRoot=fileparts(root);
    outRoot=fullfile(repoRoot,'data','incomplete_imbalanced_TIC_IMV');
    if ~exist(outRoot,'dir'), mkdir(outRoot); end
    dataName='dataset'; if isfield(S,'dataset'), dataName=char(S.dataset); end
    outputFile=fullfile(outRoot,sprintf('%s_AMR_%0.2f_seed_%d.mat', ...
        dataName,overallRate,seed));
end
outDir=fileparts(outputFile);
if ~isempty(outDir) && ~exist(outDir,'dir'), mkdir(outDir); end
save(outputFile,'-struct','S','-v7');
fprintf('Saved shared mask: target=[%s], actual=[%s], alpha=[%s]\n%s\n', ...
    sprintf(' %.4f',meta.targetMissingRate),sprintf(' %.4f',meta.actualMissingRate), ...
    sprintf(' %.4f',meta.alpha),outputFile);
end

function C=transpose_cells(X)
C=cell(size(X)); for v=1:numel(X), C{v}=X{v}'; end
end
