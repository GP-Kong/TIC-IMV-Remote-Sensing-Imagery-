function allSummary = run_TIC_IMV_protocol(datasetNames, rates)
%RUN_TIC_IMV_PROTOCOL Test TIC-IMV with Zhou et al.'s random-alpha protocol.
%
% Complete views are loaded directly from data/<dataset>.  For every
% dataset and average missing rate r in 0.1:0.1:0.5, five masks are drawn
% in memory. Each mask samples alpha_v and normalizes sum(alpha_v)=V, so
% the requested view rates are r_v=alpha_v*r. Every sample has at least
% one observed view. The same five masks are reused by every configuration,
% and each configuration is run on all five masks.
%
% This entry point does not read data/incomplete_imbalanced. It uses labels
% only for clustering metrics and oracle grid search; use a label-free
% validation protocol for final paper results.
%
% Run from MATLAB:
%   cd('D:/codex_project/CVPR/UIBAGC-main/github');
%   allSummary = run_TIC_IMV_protocol;
% To refresh only the qualitative Trento protocol while preserving the same
% five-repeat protocol and saving the pixel labels used for its metrics:
%   allSummary = run_TIC_IMV_protocol({'HSI-Lidar-Trento'},0.5);

root = fileparts(mfilename('fullpath'));
repoRoot = fileparts(root);
addpath(root);
% Use the same helper implementation as the solver and visualization.  The
% previous path referred to a non-existent checkout and made reproducibility
% depend on MATLAB's pre-existing search path.
baseFunsPath = fullfile(root,'Baselines','OSMAGC-main','funs');
if exist(baseFunsPath,'dir') == 7
    addpath(genpath(baseFunsPath),'-begin');
end

if nargin < 1 || isempty(datasetNames)
    datasetNames = {'HSI-Lidar-Trento'};
end
if nargin < 2 || isempty(rates)
    rates = 0.1:0.1:0.5;
end
datasetNames = cellstr(datasetNames);
rates = double(rates(:)');
nRepeat = 5;
maxIter = 10;
% Keep protocol files in the directory consumed by the qualitative
% visualization and the ablation/parameter scripts.
resultRoot = fullfile(root,'TIC_IMV_protocol');
if ~exist(resultRoot,'dir'), mkdir(resultRoot); end

% Same compact grid as tune_TIC_IMV.
lambdaRange = [1e-4,1e-3,1e-2];
gammaRange = [0,1e-4,1e-3];
imputedRange = [0.5,1.0];

rows = repmat(summary_template(),numel(datasetNames)*numel(rates),1);
rowCount = 0;
fprintf('TIC-IMV protocol: %d datasets x %d rates x %d repeats.\n', ...
    numel(datasetNames),numel(rates),nRepeat);

for di = 1:numel(datasetNames)
    source = load_raw_dataset(datasetNames{di},fullfile(repoRoot,'data'));
    V = source.V;
    if V == 2
        anchorGrid = [50 50;50 100;100 50;100 100];
    elseif V == 3
        anchorGrid = [50 50 50;50 100 100;100 50 100;100 100 100];
    else
        error('Only two- and three-view datasets are supported; V=%d.',V);
    end
    fprintf('\n=== %s: N=%d, V=%d, classes=%d ===\n', ...
        source.dataset,source.N,source.V,number_of_classes(source.truelabel));

    for ri = 1:numel(rates)
        r = rates(ri);
        scenarios = cell(nRepeat,1);
        scenarioMasks = cell(nRepeat,1);
        scenarioSeeds = zeros(nRepeat,1);
        actualRates = zeros(nRepeat,V);
        alphaValues = zeros(nRepeat,V);
        globalRates = zeros(nRepeat,1);
        for rep = 1:nRepeat
            seed = 100000*di+1000*ri+rep;
            scenarios{rep} = make_protocol_scenario(source,r,rep,seed);
            scenarioMasks{rep} = scenarios{rep}.MissingStatus;
            scenarioSeeds(rep) = seed;
            actualRates(rep,:) = scenarios{rep}.actualMissingRate;
            alphaValues(rep,:) = scenarios{rep}.alpha;
            globalRates(rep) = scenarios{rep}.averageMissingRate;
        end
        fprintf('[%s, r=%.1f] global mean=%.4f, view rates mean=[%s]\n', ...
            source.dataset,r,mean(globalRates),sprintf(' %.4f',mean(actualRates,1)));
        [bestCfg,bestRep,bestFinal,grid] = tune_protocol_rate( ...
            scenarios,anchorGrid,lambdaRange,gammaRange,imputedRange, ...
            maxIter,resultRoot,source.dataset,r);
        rateSummary = struct();
        rateSummary.dataset = source.dataset;
        rateSummary.averageMissingRate = r;
        rateSummary.repeatCount = nRepeat;
        rateSummary.alpha = alphaValues;
        rateSummary.scenarioSeeds = scenarioSeeds;
        rateSummary.masks = scenarioMasks;
        rateSummary.actualMissingRate = actualRates;
        rateSummary.averageMissingRateRealized = globalRates;
        rateSummary.bestConfig = bestCfg;
        rateSummary.bestIteration = bestRep.bestIter;
        rateSummary.bestResults = bestRep.results;
        % Save the raw k-means labels used to compute bestResults.  The
        % previous protocol format kept only metrics, which made an exact
        % qualitative-map reconstruction impossible after code changes.
        rateSummary.bestPredLabels = bestRep.predLabels;
        rateSummary.predictionLabelConvention = ...
            'cell{repeat}: N-by-1 raw cluster IDs at bestIteration';
        rateSummary.finalResults = bestFinal;
        rateSummary.meanBestResult = mean(bestRep.results,1);
        rateSummary.stdBestResult = std(bestRep.results,0,1);
        rateSummary.meanFinalResult = mean(bestFinal,1);
        rateSummary.stdFinalResult = std(bestFinal,0,1);
        rateSummary.grid = grid;
        rateSummary.sourceDirectory = source.sourceDirectory;
        rateSummary.sourceDescription = source.sourceDescription;
        rateSummary.maskConvention = 'N-by-V; 1 observed, 0 missing';
        rateSummary.protocolCode = 'TIC-IMV local solver with stored bestPredLabels';
        rateSummary.protocolGenerated = datestr(now,30);
        rateFile = fullfile(resultRoot,sprintf('%s_r_%0.1f.mat',source.dataset,r));
        save(rateFile,'rateSummary','-v7');
        write_rate_csv(rateSummary,fullfile(resultRoot,sprintf( ...
            '%s_r_%0.1f.csv',source.dataset,r)));
        rowCount = rowCount+1;
        rows(rowCount) = summary_row(rateSummary);
    end
end

allSummary = rows(1:rowCount);
% A targeted refresh (for example, Trento at r=0.5) must not overwrite the
% aggregate file from a previous full four-dataset protocol run.
isFullProtocol = numel(datasetNames) == 4 && numel(rates) == 5;
if isFullProtocol
    summaryBase = 'TIC_IMV_protocol_all_summary';
else
    summaryBase = 'TIC_IMV_protocol_subset_summary';
end
save(fullfile(resultRoot,[summaryBase '.mat']),'allSummary', ...
    'datasetNames','rates','nRepeat','-v7');
write_all_csv(allSummary,fullfile(resultRoot,[summaryBase '.csv']));
fprintf('\nFinished. Results: %s\n',resultRoot);
end

function source = load_raw_dataset(name,dataRoot)
%LOAD_RAW_DATASET Read a complete dataset from its original data folder.
% Every returned view is D-by-N. Houston2018 uses only the raw coordinate
% list to make the very large scene tractable; features and labels still
% come from raw HSI.mat, LiDAR.mat and gt.mat.
source = struct('dataset',name,'sourceDirectory','','sourceDescription','', ...
    'data_full',{{}},'truelabel',[],'gt',[],'imageSize',[],'V',0,'N',0, ...
    'viewNames',{{}},'evalMask',[]);
switch lower(name)
    case 'augsburg3m'
        folder = fullfile(dataRoot,'Augsburg3M');
        H = load(fullfile(folder,'data_HS_LR.mat'),'data_HS_LR');
        R = load(fullfile(folder,'data_SAR_HR.mat'),'data_SAR_HR');
        D = load(fullfile(folder,'data_DSM.mat'),'data_DSM');
        G = load(fullfile(folder,'Augsburg_gt.mat'),'Augsburg_gt');
        source.data_full = {to_features(H.data_HS_LR), ...
            to_features(R.data_SAR_HR),to_features(D.data_DSM)};
        source.truelabel = double(G.Augsburg_gt(:));
        source.imageSize = size(G.Augsburg_gt);
        source.viewNames = {'HSI180','SAR4','DSM1'};
        source.sourceDescription = 'Raw Augsburg3M files.';
    case 'houston2018'
        folder = fullfile(dataRoot,'Houston2018');
        H = load(fullfile(folder,'HSI.mat'),'HSI');
        L = load(fullfile(folder,'LiDAR.mat'),'LiDAR');
        G = load(fullfile(folder,'gt.mat'),'gt');
        I = load(fullfile(folder,'houston_index.mat'),'houston_test_small');
        [h,w,~] = size(H.HSI);
        ij = double(I.houston_test_small);
        if isempty(ij) || size(ij,2) ~= 2
            error('houston_test_small must be an N-by-2 coordinate array.');
        end
        % The distributed index is zero-based; convert to MATLAB subscripts.
        lin = sub2ind([h,w],ij(:,1)+1,ij(:,2)+1);
        labels = double(G.gt(lin));
        valid = labels > 0 & isfinite(labels);
        lin = lin(valid); labels = labels(valid);
        H2 = reshape(H.HSI,h*w,size(H.HSI,3));
        L2 = reshape(L.LiDAR,h*w,[]);
        source.data_full = {double(H2(lin,:))',double(L2(lin,:))'};
        source.truelabel = labels(:);
        source.imageSize = [];
        source.viewNames = {'HSI50','LiDAR1'};
        source.sourceDescription = ['Raw Houston2018 files sampled at ', ...
            'houston_test_small coordinates.'];
    case 'muufl'
        folder = fullfile(dataRoot,'MUUFL');
        H = load(fullfile(folder,'HSI.mat'),'HSI');
        L = load(fullfile(folder,'LiDAR.mat'),'LiDAR');
        G = load(fullfile(folder,'gt.mat'),'gt');
        source.data_full = {to_features(H.HSI),to_features(L.LiDAR)};
        source.truelabel = double(G.gt(:));
        source.imageSize = size(G.gt);
        source.viewNames = {'HSI64','LiDAR2'};
        source.sourceDescription = 'Raw MUUFL files.';
    case 'hsi-lidar-trento'
        folder = fullfile(dataRoot,'HSI-Lidar-Trento');
        H = load(fullfile(folder,'Trento-HSI.mat'),'HSI');
        L = load(fullfile(folder,'Trento-Lidar.mat'),'Lidar');
        G = load(fullfile(folder,'Trento-GT.mat'),'GT');
        source.data_full = {to_features(H.HSI),to_features(L.Lidar)};
        source.truelabel = double(G.GT(:));
        source.imageSize = size(G.GT);
        source.viewNames = {'HSI63','LiDAR2'};
        source.sourceDescription = 'Raw HSI-LiDAR Trento files.';
    otherwise
        error('Unsupported raw dataset: %s',name);
end
source.gt = source.truelabel;
source.V = numel(source.data_full);
source.N = numel(source.truelabel);
source.evalMask = source.truelabel > 0;
for v = 1:source.V
    if size(source.data_full{v},2) ~= source.N
        error('%s view %d has N=%d; labels have N=%d.',name,v, ...
            size(source.data_full{v},2),source.N);
    end
end
source.sourceDirectory = folder;
end

function S = make_protocol_scenario(source,r,rep,seed)
%MAKE_PROTOCOL_SCENARIO Make one in-memory shared random-alpha scenario.
[M,meta] = make_shared_missing_mask(source.N,source.V,r,seed,'tic_imv');
S = source;
S.scenario = sprintf('%s_AMR_%.1f_rep_%02d',source.dataset,r,rep);
S.data = source.data_full;
for v = 1:source.V, S.data{v}(:,~M(:,v)') = NaN; end
S.MissingStatus = uint8(M);
S.missingMask = uint8(~M);
S.targetMissingRate = meta.targetMissingRate;
S.actualMissingRate = meta.actualMissingRate;
S.averageMissingRate = meta.averageMissingRate;
S.observationRate = meta.observationRate;
S.index = cell(1,source.V);
for v = 1:source.V, S.index{v} = find(M(:,v)); end
S.N = source.N; S.V = source.V; S.seed = seed; S.rep = rep;
S.alpha = meta.alpha; S.maskMeta = meta;
S.maskProtocol = 'TIC-IMV random alpha; one shared mask for all methods';
end

function [bestCfg,bestRep,bestFinal,grid] = tune_protocol_rate(scenarios, ...
    anchorGrid,lambdaRange,gammaRange,imputedRange,maxIter,resultRoot, ...
    datasetName,r)
total = size(anchorGrid,1)*numel(lambdaRange)*numel(gammaRange)*numel(imputedRange);
grid = repmat(struct('score',NaN,'stdACC',NaN,'anchorNum',[], ...
    'lambda',NaN,'gamma',NaN,'imputedConsensusWeight',NaN,'successCount',0), ...
    total,1);
bestScore = -inf;
bestCfg = struct();
bestRep = struct('results',nan(numel(scenarios),6), ...
    'bestIter',nan(numel(scenarios),1), ...
    'predLabels',{cell(numel(scenarios),1)});
bestFinal = nan(numel(scenarios),6);
count = 0;
for ai = 1:size(anchorGrid,1)
    for li = 1:numel(lambdaRange)
        for gi = 1:numel(gammaRange)
            for mi = 1:numel(imputedRange)
                count = count+1;
                opts = struct('anchorNum',anchorGrid(ai,:), ...
                    'lambda',lambdaRange(li),'gamma',gammaRange(gi), ...
                    'imputedConsensusWeight',imputedRange(mi), ...
                    'maxIter',maxIter,'verbose',false,'saveResults',false, ...
                    'outputDir',fullfile(resultRoot,'_protocol_tmp'));
                repResults = nan(numel(scenarios),6);
                repFinal = nan(numel(scenarios),6);
                repIter = nan(numel(scenarios),1);
                repLabels = cell(numel(scenarios),1);
                for rep = 1:numel(scenarios)
                    try
                        [summary,model] = main_TIC_IMV_incomplete(scenarios{rep},opts);
                        repIter(rep) = summary.bestIter;
                        repResults(rep,:) = summary.allResults(summary.bestIter,:);
                        repFinal(rep,:) = summary.finalResult;
                        repLabels{rep} = raw_labels_at_iteration( ...
                            model,opts,repIter(rep));
                    catch ME
                        fprintf(2,'Failed %s r=%.1f config %d repeat %d: %s\n', ...
                            datasetName,r,count,rep,ME.message);
                    end
                end
                ok = all(isfinite(repResults(:,1)));
                if ok
                    score = mean(repResults(:,1)); spread = std(repResults(:,1),0,1);
                else
                    score = NaN; spread = NaN;
                end
                grid(count).score = score; grid(count).stdACC = spread;
                grid(count).anchorNum = anchorGrid(ai,:);
                grid(count).lambda = opts.lambda; grid(count).gamma = opts.gamma;
                grid(count).imputedConsensusWeight = opts.imputedConsensusWeight;
                grid(count).successCount = sum(isfinite(repResults(:,1)));
                fprintf(['%s r=%.1f config %d/%d A=[%s] lambda=%.1e ', ...
                    'gamma=%.1e imputed=%.2g meanBestACC=%.4f\n'], ...
                    datasetName,r,count,total,sprintf('%d ',opts.anchorNum), ...
                    opts.lambda,opts.gamma,opts.imputedConsensusWeight,score);
                if ok && score > bestScore
                    bestScore = score; bestCfg = opts;
                    bestCfg.tuningScore = score; bestCfg.tuningStdACC = spread;
                    bestCfg.dataset = datasetName; bestCfg.averageMissingRate = r;
                    bestRep.results = repResults; bestRep.bestIter = repIter;
                    bestRep.predLabels = repLabels;
                    bestFinal = repFinal;
                end
            end
        end
    end
end
if ~isfinite(bestScore)
    error('No parameter configuration completed all %d repeats.',numel(scenarios));
end
end

function labels = raw_labels_at_iteration(model,~,iter)
%RAW_LABELS_AT_ITERATION Reproduce the labels used by main_TIC_IMV... metrics.
if iter < 1 || iter > numel(model.Hlist)
    error('Requested prediction iteration %d is unavailable.',iter);
end
% litekmeans returns the sample labels as its first output and the cluster
% centers as its second output.  The previous version saved the centers,
% which made bestPredLabels contain K-by-K matrices instead of N pixel labels.
rng(double(model.options.seed)+double(iter),'twister');
[labels,~] = litekmeans(model.Hlist{iter}',model.clusterNum);
labels = double(labels(:));
end

function row = summary_template()
row = struct('dataset','','averageMissingRate',NaN,'bestACC',NaN, ...
    'bestKappa',NaN,'bestNMI',NaN,'bestPurity',NaN,'bestARI',NaN, ...
    'bestF1',NaN,'bestACCStd',NaN,'bestKappaStd',NaN,'bestNMIStd',NaN, ...
    'bestPurityStd',NaN,'bestARIStd',NaN,'bestF1Std',NaN,'anchorNum',[], ...
    'lambda',NaN,'gamma',NaN,'imputedConsensusWeight',NaN, ...
    'actualMissingRate',[]);
end

function row = summary_row(S)
row = struct('dataset',S.dataset,'averageMissingRate',S.averageMissingRate, ...
    'bestACC',S.meanBestResult(1),'bestKappa',S.meanBestResult(2), ...
    'bestNMI',S.meanBestResult(3),'bestPurity',S.meanBestResult(4), ...
    'bestARI',S.meanBestResult(5),'bestF1',S.meanBestResult(6), ...
    'bestACCStd',S.stdBestResult(1),'bestKappaStd',S.stdBestResult(2), ...
    'bestNMIStd',S.stdBestResult(3),'bestPurityStd',S.stdBestResult(4), ...
    'bestARIStd',S.stdBestResult(5),'bestF1Std',S.stdBestResult(6), ...
    'anchorNum',S.bestConfig.anchorNum,'lambda',S.bestConfig.lambda, ...
    'gamma',S.bestConfig.gamma,'imputedConsensusWeight', ...
    S.bestConfig.imputedConsensusWeight,'actualMissingRate', ...
    mean(S.actualMissingRate,1));
end

function write_rate_csv(S,fileName)
fid = fopen(fileName,'w');
if fid < 0, error('Cannot write %s',fileName); end
fprintf(fid,['dataset,averageMissingRate,repeat,bestIter,ACC,Kappa,NMI,Purity,ARI,F1,', ...
    'finalACC,finalKappa,finalNMI,finalPurity,finalARI,finalF1,', ...
    'actualMissingRateV1,actualMissingRateV2,actualMissingRateV3\n']);
for i = 1:S.repeatCount
    z = nan(1,3); z(1:min(3,size(S.actualMissingRate,2))) = ...
        S.actualMissingRate(i,1:min(3,size(S.actualMissingRate,2)));
    fprintf(fid,['%s,%.1f,%d,%d,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,', ...
        '%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f\n'], ...
        S.dataset,S.averageMissingRate,i,S.bestIteration(i), ...
        [S.bestResults(i,:),S.finalResults(i,:),z]);
end
fprintf(fid,['MEAN,%.1f,0,0,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,', ...
    '%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f\n'], ...
    S.averageMissingRate,[S.meanBestResult,S.meanFinalResult,mean(S.actualMissingRate,1)]);
fprintf(fid,['STD,%.1f,0,0,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,', ...
    '%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f\n'], ...
    S.averageMissingRate,[S.stdBestResult,S.stdFinalResult,std(S.actualMissingRate,0,1)]);
fclose(fid);
end

function write_all_csv(rows,fileName)
fid = fopen(fileName,'w');
if fid < 0, error('Cannot write %s',fileName); end
fprintf(fid,['dataset,averageMissingRate,bestACC,bestKappa,bestNMI,bestPurity,', ...
    'bestARI,bestF1,bestACCStd,bestKappaStd,bestNMIStd,bestPurityStd,', ...
    'bestARIStd,bestF1Std,anchor1,anchor2,anchor3,lambda,gamma,', ...
    'imputedConsensusWeight,actualMissingRateV1,actualMissingRateV2,actualMissingRateV3\n']);
for i = 1:numel(rows)
    a = nan(1,3); a(1:min(3,numel(rows(i).anchorNum))) = rows(i).anchorNum;
    m = nan(1,3); m(1:min(3,numel(rows(i).actualMissingRate))) = rows(i).actualMissingRate;
    fprintf(fid,['%s,%.1f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,', ...
        '%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%g,%g,%g,%.8g,%.8g,', ...
        '%.8g,%.8f,%.8f,%.8f\n'],rows(i).dataset, ...
        rows(i).averageMissingRate,rows(i).bestACC,rows(i).bestKappa, ...
        rows(i).bestNMI,rows(i).bestPurity,rows(i).bestARI,rows(i).bestF1, ...
        rows(i).bestACCStd,rows(i).bestKappaStd,rows(i).bestNMIStd, ...
        rows(i).bestPurityStd,rows(i).bestARIStd,rows(i).bestF1Std,a, ...
        rows(i).lambda,rows(i).gamma,rows(i).imputedConsensusWeight,m);
end
fclose(fid);
end

function X = to_features(A)
% Convert H-by-W-by-D or H-by-W arrays to D-by-N matrices.
A = double(A);
if ndims(A) <= 2, X = reshape(A,1,[]);
else, X = reshape(A,[],size(A,3))'; end
end

function K = number_of_classes(labels)
u = unique(double(labels(:))); u = u(u > 0 & isfinite(u));
K = numel(u);
if K < 2, error('At least two positive classes are required.'); end
end
