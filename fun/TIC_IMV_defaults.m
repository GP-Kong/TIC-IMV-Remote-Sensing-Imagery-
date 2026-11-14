function opts = TIC_IMV_defaults()
%TIC_IMV_DEFAULTS Defaults for the tensor-free balanced OSMAGC solver.
% The solver keeps the OSMAGC anchor graph and orthogonal alignment blocks.
% The extra terms are sample-mask aware and do not use ground-truth labels.
opts.anchorNum = 50;
opts.anchorInitMode = 'osmagc';
opts.anchorLabels = {};
opts.clusterNum = [];
opts.maxIter = 10;
opts.tol = 1e-4;
opts.lambda = 1e-3;             % alignment between view graphs and H
opts.gamma = 1e-3;              % spatial smoothness when imageSize is known
opts.neighborNum = [];           % default: clusterNum nearest anchors
opts.seed = 1;
opts.verbose = true;

% Observation-pattern imbalance compensation.
opts.missingExponent = 0.10;     % only a mild training-time observation correction
opts.minViewWeight = 0.75;
opts.maxViewWeight = 1.25;
opts.residualExponent = 0.25;
opts.weightMomentum = 0.90;      % avoid late-iteration consensus drift

% TIC-IMV-inspired contribution and distribution balancing.
opts.useContributionBalance = true;
opts.contributionExponent = 0.10;
opts.contributionTemperature = 1.0;
opts.useDistributionBalance = true;
opts.distributionExponent = 0.10;
opts.distributionTemperature = 1.0;
opts.curriculumStrength = 0.10;
opts.curriculumWarmup = 0.30;

% CIMLN-inspired counterfactual view masking (enabled in the proposed model).
opts.interventionProb = 0.15;
opts.invariantBlend = 0.10;
% OSMAGC reconstructs missing-view anchor coefficients from H and includes
% those coefficients in the consensus update.  Keep that cross-view signal;
% the tuning script also evaluates 0 and 0.5 for noisier datasets.
opts.imputedConsensusWeight = 1.0;
% Use adaptive weights for the view-specific Z updates, but keep the final
% consensus equal-weighted by default.  This matches OSMAGC's stable H block
% and prevents a noisy low-observation view from feeding back into H.
opts.consensusWeightMode = 'uniform';
opts.alignmentMaskMode = 'all';
opts.initConsensusMode = 'osmagc';
% 'balanced_residual' follows OSMAGC's inverse residual weighting and lets
% the imbalance controller make only a bounded secondary correction.
opts.zWeightMode = 'balanced_residual';
opts.zImbalanceScale = 0.15;
opts.alphaMomentum = 0;
opts.imbalanceMode = 'light';
opts.consensusBlend = 1.0;
opts.imputedConfidenceMode = 'none';
opts.imputedConfidenceFloor = 0.25;
opts.saveResults = true;

% Fixed BMvC-inspired view update balancing used by the proposed model.
opts.vcrExponent = 0.50;
opts.vcrMinScale = 0.50;
opts.vcrMaxScale = 2.00;

% Fixed conservative unsupervised soft class-frequency correction.
opts.classExponent = 0.20;
opts.classBlend = 0.05;
opts.classWeightMin = 0.50;
opts.classWeightMax = 2.00;
opts.classPriorMode = 'uniform';
opts.classPriorEMA = 0.90;
opts.classPriorMaxRatio = 20;

% Anchor graph / initialization.
opts.initReplicates = 1;
opts.initMaxIter = 50;
opts.imageSize = [];
opts.returnLabels = true;
end
