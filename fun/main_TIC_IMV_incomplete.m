function [summary, model] = main_TIC_IMV_incomplete(scenarioName, userOpts)
%MAIN_TIC_IMV_INCOMPLETE Run the tensor-free model on one shared split.
% Example: main_TIC_IMV_incomplete('Houston2013_mr_0p1_0p7');
% The random-alpha raw-data protocol may also pass an in-memory scenario
% struct containing data_full, data, truelabel, MissingStatus and imageSize.
if nargin<1 || isempty(scenarioName)
    scenarioName='Augsburg3M_mr_0p1_0p5_0p8';
end
if nargin<2, userOpts=struct(); end
root=fileparts(mfilename('fullpath')); repoRoot=fileparts(root);
dataRoot=fullfile(repoRoot,'data','incomplete_imbalanced');
% The protocol runner passes an in-memory struct made from the complete raw
% dataset.  Keep the file-name interface for the existing prepared-scenario
% scripts, but do not force a struct through char()/load().
if isstruct(scenarioName)
    S=scenarioName;
    if isfield(S,'scenario') && ~isempty(S.scenario)
        scenarioName=char(S.scenario);
    elseif isfield(S,'dataset')
        scenarioName=char(S.dataset);
    else
        scenarioName='in_memory_scenario';
    end
    scenarioFile='';
else
    scenarioArg=char(scenarioName);
    if exist(scenarioArg,'file')
        scenarioFile=scenarioArg;
        [~,scenarioName]=fileparts(scenarioFile);
    else
        scenarioFile=fullfile(dataRoot,[scenarioArg '.mat']);
    end
    if ~exist(scenarioFile,'file'), error('Scenario file not found: %s',scenarioFile); end
    S=load(scenarioFile);
end
addpath(root);
% Keep every protocol and visualization run on the same local helper
% implementation.  The old path pointed to a repository layout that is not
% present in this project and could silently resolve litekmeans/preData from
% whichever unrelated baseline happened to be on MATLAB's path.
localFuns=fullfile(root,'Baselines','OSMAGC-main','funs');
if exist(localFuns,'dir')
    addpath(genpath(localFuns),'-begin');
end
evalDir=fullfile(repoRoot,'ETR-OSMAGC');
if exist(evalDir,'dir')
    % The evaluator is shared with the ETR code, but it must come after the
    % OSMAGC helper directory so that litekmeans is deterministic here.
    addpath(evalDir,'-end');
end

if ~isfield(S,'data'), error('Scenario must contain data.'); end
X=S.data;
if isfield(S,'MissingStatus'), M=logical(S.MissingStatus);
elseif isfield(S,'index')
    M=false(S.N,S.V);
    for v=1:S.V, M(S.index{v},v)=true; end
else
    M=false(size(X{1},2),numel(X));
    for v=1:numel(X), M(:,v)=all(isfinite(X{v}),1)'; end
end
if isfield(S,'truelabel'), gt=double(S.truelabel(:));
elseif isfield(S,'gt'), gt=double(S.gt(:)); else, error('No labels in scenario.'); end
K=number_of_classes(gt);
opts=TIC_IMV_defaults(); opts.clusterNum=K;
% Keep defaults unless the caller supplies a value.  The previous version
% overwrote maxIter/anchorNum here, making parameter sweeps ineffective.
if ~isfield(userOpts,'anchorNum'), opts.anchorNum=50; end
if ~isfield(userOpts,'maxIter'), opts.maxIter=10; end
if ~isfield(userOpts,'lambda'), opts.lambda=1e-3; end
if ~isfield(userOpts,'gamma'), opts.gamma=1e-3; end
if isfield(S,'imageSize') && numel(S.imageSize)==2 && ...
        prod(double(S.imageSize))==numel(gt)
    opts.imageSize=double(S.imageSize(:)');
else
    opts.imageSize=[];
end
opts=merge_local(opts,userOpts);
if isfield(S,'seed') && ~isfield(userOpts,'seed'), opts.seed=double(S.seed); end
% Reuse the OSMAGC ERS partition when a complete image grid and mex_ers are
% available. The solver still computes each anchor only from observed samples.
if isempty(opts.anchorLabels) && ~isempty(opts.imageSize) && ...
        exist('preData','file')==2 && exist('mex_ers','file')==3
    try
        cube=cell(1,numel(X));
        for v=1:numel(X)
            cube{v}=reshape(double(S.data_full{v})',opts.imageSize(1), ...
                opts.imageSize(2),size(S.data_full{v},1));
        end
        anchorCounts=opts.anchorNum;
        if isscalar(anchorCounts), anchorCounts=repmat(anchorCounts,1,numel(X)); end
        splabels=preData(cube,anchorCounts,numel(X));
        opts.anchorLabels=cell(1,numel(X));
        for v=1:numel(X), opts.anchorLabels{v}=double(splabels{v}(:)); end
    catch ME
        fprintf('ERS anchor initialization unavailable: %s\n',ME.message);
        opts.anchorLabels={};
    end
end
% Irregular labeled-pixel scenarios (Houston2013/Houston2018) do not have
% a complete image grid.  Match OSMAGC's feature-space anchor initialization
% instead of falling back to a different random anchor construction.
if isempty(opts.anchorLabels) && isfield(S,'data_full') && ...
        exist('make_feature_anchor_labels','file')==2
    try
        anchorCounts=opts.anchorNum;
        if isscalar(anchorCounts), anchorCounts=repmat(anchorCounts,1,numel(X)); end
        opts.anchorLabels=make_feature_anchor_labels(S.data_full,numel(X), ...
            anchorCounts,numel(gt));
    catch ME
        fprintf('Feature anchor initialization unavailable: %s\n',ME.message);
        opts.anchorLabels={};
    end
end

fprintf('TIC-IMV: %s, N=%d, V=%d, target/actual missing=[%s]/[%s]\n', ...
    scenarioName,numel(gt),numel(X),rate_text(S,'targetMissingRate'), ...
    rate_text(S,'actualMissingRate'));
tic; [~,model]=TIC_IMV(X,M,opts); elapsed=toc;
valid=gt>0 & isfinite(gt); Hlist=model.Hlist; nIter=numel(Hlist);
computeMetrics = true;
if isfield(opts,'computeMetrics'), computeMetrics = logical(opts.computeMetrics); end
if computeMetrics
    results=zeros(nIter,6);
    for it=1:nIter
        rng(opts.seed+it,'twister');
        pred=litekmeans(Hlist{it}',K); pred=pred(:);
        results(it,:)=evaluate_results_clustering(gt(valid),pred(valid));
    end
    [bestACC,bestIter]=max(results(:,1)); finalResult=results(end,:);
else
    results=zeros(nIter,0);
    bestACC=NaN; bestIter=nIter; finalResult=[];
end
summaryOpts=opts; summaryOpts.anchorLabels={};
if isfield(opts,'outputDir') && ~isempty(opts.outputDir)
    outRoot=opts.outputDir;
else
    outRoot=fullfile(root,'results','incomplete_imbalanced');
end
summary=struct('dataset',get_field(S,'dataset',scenarioName), ...
    'scenario',get_field(S,'scenario',''),'scenarioName',scenarioName, ...
    'seed',get_field(S,'seed',opts.seed),'actualMissingRate',actual_rate(M), ...
    'targetMissingRate',get_field(S,'targetMissingRate',[]),'bestACC',bestACC, ...
    'bestIter',bestIter,'finalResult',finalResult,'allResults',results, ...
    'elapsedSeconds',elapsed,'options',summaryOpts, ...
    'viewWeightHistory',model.viewWeightHistory, ...
    'viewContributionHistory',model.viewContributionHistory, ...
    'viewOTBiasHistory',model.viewOTBiasHistory, ...
    'objectiveHistory',model.objectiveHistory);
if ~isfield(opts,'saveResults') || opts.saveResults
    if ~exist(outRoot,'dir'), mkdir(outRoot); end
    save(fullfile(outRoot,[scenarioName '_TIC-IMV.mat']),'summary','-v7');
    csvFile=fullfile(outRoot,[scenarioName '_TIC-IMV.csv']); fid=fopen(csvFile,'w');
    if fid<0, error('Cannot open result file for writing: %s',csvFile); end
    fprintf(fid,'iter,ACC,Kappa,NMI,Purity,ARI,F1\n');
    for it=1:nIter
        fprintf(fid,'%d,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f\n',it,results(it,:));
    end
    fclose(fid);
end
if computeMetrics
    fprintf('Best ACC %.4f at iter %d; final ACC %.4f; time %.2fs\n', ...
        bestACC,bestIter,finalResult(1),elapsed);
else
    fprintf('Objective-only run: %d iterations; time %.2fs\n',nIter,elapsed);
end
end

function K=number_of_classes(gt)
u=unique(gt(gt>0 & isfinite(gt))); K=numel(u);
if K<2, error('At least two positive classes are required.'); end
end
function out=merge_local(base,user)
out=base; if isempty(user), return; end
fn=fieldnames(user); for i=1:numel(fn), out.(fn{i})=user.(fn{i}); end
end
function x=get_field(S,name,default)
if isfield(S,name), x=S.(name); else, x=default; end
end
function txt=rate_text(S,name)
if isfield(S,name), txt=sprintf(' %.4f',double(S.(name))); else, txt=' n/a'; end
end
function r=actual_rate(M), r=1-mean(double(M),1); end
