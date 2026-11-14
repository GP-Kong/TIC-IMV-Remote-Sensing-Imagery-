function [H, model] = TIC_IMV(X, M, opts)
%TIC_IMV Tensor-free, imbalance-aware OSMAGC for incomplete views.
% X: 1-by-V cell array; each X{v} is D_v-by-N. Missing columns may be NaN.
% M: N-by-V logical observation mask (1 observed, 0 missing).
% The model retains OSMAGC's anchor graph, orthogonal alignment and spatial
% consensus, and adds shallow mask/contribution/update balancing terms.

if nargin < 2 || isempty(M)
    M = false(size(X{1},2), numel(X));
    for v = 1:numel(X), M(:,v) = all(isfinite(X{v}),1)'; end
end
if nargin < 3 || isempty(opts), opts = TIC_IMV_defaults();
else, opts = merge_options(TIC_IMV_defaults(), opts); end
if opts.maxIter<1, error('opts.maxIter must be at least 1.'); end

V = numel(X);
if V < 1, error('X must contain at least one view.'); end
N = size(X{1},2);
if ~isequal(size(M), [N,V]), error('M must be N-by-V and match X.'); end
if any(sum(logical(M),2) == 0)
    error('Every sample must have at least one observed view.');
end
M = logical(M);
rng(opts.seed, 'twister');

K = opts.clusterNum;
if isempty(K), K = min(10, max(2, round(sqrt(N/2)))); end
anchorNum = opts.anchorNum;
if isscalar(anchorNum), anchorNum = repmat(anchorNum,1,V); end
if numel(anchorNum) ~= V, error('anchorNum must be scalar or length V.'); end

% Normalize observed columns only; missing entries remain exact zeros.
for v = 1:V
    if size(X{v},2) ~= N, error('All views must have N columns.'); end
    X{v} = double(X{v});
    X{v}(~isfinite(X{v})) = 0;
    X{v} = normalize_columns(X{v});
end

obsRate = mean(M,1);
A = cell(1,V); Z = cell(1,V); R = cell(1,V);
for v = 1:V
    idx = find(M(:,v));
    if isempty(idx), error('View %d has no observed samples.',v); end
    m = max(2, min(round(anchorNum(v)), numel(idx)));
    initIdx=idx;
    if isfield(opts,'anchorInitMode') && strcmpi(opts.anchorInitMode,'osmagc')
        % The OSMAGC baseline forms anchors before applying q_v.  Using all
        % samples here makes the comparison fair while A/Z updates remain
        % mask-aware afterward.
        initIdx=1:N;
    end
    if numel(opts.anchorLabels)>=v && ~isempty(opts.anchorLabels{v})
        labels=double(opts.anchorLabels{v}(:));
        if numel(labels)==N
            A{v}=init_anchors_from_labels(X{v}(:,initIdx),labels(initIdx),m,opts);
        else
            A{v}=init_anchors(X{v}(:,initIdx),m,opts);
        end
    else
        A{v} = init_anchors(X{v}(:,initIdx), m, opts);
    end
    Z{v} = zeros(m,N);
    if isempty(opts.neighborNum), nn = min(K,m-1); else, nn = opts.neighborNum; end
    Z{v}(:,idx) = anchor_graph(X{v}(:,idx), A{v}, nn);
    R{v} = eye(K,m);
    if K ~= m
        [u,~,vv] = svd(randn(K,m),'econ'); R{v} = u*vv';
    end
end
if isfield(opts,'initConsensusMode') && strcmpi(opts.initConsensusMode,'osmagc')
    H = init_osmagc_consensus(K,N);
else
    H = init_consensus(Z,R,M,K,N);
end
spatialD = [];
if numel(opts.imageSize)==2 && prod(opts.imageSize)==N
    spatialD = grid_laplacian(opts.imageSize);
end

viewWeight = initial_weights(obsRate, opts);
alpha = ones(1,V)/V;
history = zeros(opts.maxIter,4);
weightHistory = zeros(opts.maxIter,V);
contributionHistory = zeros(opts.maxIter,V);
otBiasHistory = zeros(opts.maxIter,V);
updateScaleHistory = ones(opts.maxIter,V);
alphaHistory = zeros(opts.maxIter,V);
% Label-free iteration-wise surrogate objective for convergence analysis.
objectiveHistory = nan(opts.maxIter,1);
Hlist = cell(1,opts.maxIter);
prevH = H;
for it = 1:opts.maxIter
    % A: observed-only orthogonal reconstruction update.
    for v=1:V
        idx=find(M(:,v));
        A{v}=orthogonal_update(X{v}(:,idx),Z{v}(:,idx),A{v});
    end
    % R: use completed anchor coefficients for alignment.  OSMAGC updates R
    % with all samples; restricting it to observed columns makes a heavily
    % missing view unable to align its completed graph to the consensus.
    for v=1:V
        if isfield(opts,'alignmentMaskMode') && ...
                strcmpi(opts.alignmentMaskMode,'observed')
            idx=find(M(:,v));
            R{v}=orthogonal_update(H(:,idx),Z{v}(:,idx),R{v});
        else
            R{v}=orthogonal_update(H,Z{v},R{v});
        end
    end
    % Z: preserve the OSMAGC simplex anchor graph update, with bounded
    % per-view reconstruction weights and a VCR-style step-size balance.
    Zcandidate=Z; updateNorm=zeros(1,V);
    for v=1:V
        idx=find(M(:,v));
        if isfield(opts,'zWeightMode') && strcmpi(opts.zWeightMode,'uniform')
            recWeight=1;
        elseif isfield(opts,'zWeightMode') && strcmpi(opts.zWeightMode,'osmagc')
            recWeight=1/max(alpha(v),1e-8);
        else
            % Preserve OSMAGC's residual-derived alpha update, then apply
            % the bounded mask/contribution factor as a secondary correction.
            imbalanceScale=max(opts.zImbalanceScale,0);
            recWeight=(1/max(alpha(v),1e-8))* ...
                (1+imbalanceScale*(viewWeight(v)-1));
        end
        F=recWeight*(A{v}'*X{v}(:,idx)) + opts.lambda*(R{v}'*H(:,idx));
        den=recWeight+opts.lambda;
        Zcandidate{v}(:,idx)=project_columns(bsxfun(@rdivide,F,den));
        updateNorm(v)=norm(Zcandidate{v}(:,idx)-Z{v}(:,idx),'fro') / ...
            max(norm(Z{v}(:,idx),'fro'),1e-10);
        miss=find(~M(:,v));
        if ~isempty(miss)
            Zcandidate{v}(:,miss)=project_columns(R{v}'*H(:,miss));
        end
    end
    scales=ones(1,V);
    valid=updateNorm>1e-10;
    if any(valid)
        target=median(updateNorm(valid));
        scales(valid)=(target./updateNorm(valid)).^opts.vcrExponent;
        scales=min(max(scales,opts.vcrMinScale),opts.vcrMaxScale);
    end
    for v=1:V
        idx=find(M(:,v));
        Z{v}(:,idx)=project_columns(Z{v}(:,idx)+ ...
            scales(v)*(Zcandidate{v}(:,idx)-Z{v}(:,idx)));
        miss=find(~M(:,v)); Z{v}(:,miss)=Zcandidate{v}(:,miss);
    end

    % View-specific aligned memberships and weighted OSMAGC consensus.
    Sview=cell(1,V); Sprob=cell(1,V);
    for v=1:V
        Sview{v}=R{v}*Z{v};
        Sprob{v}=project_columns(Sview{v});
    end
    imputedConfidence=compute_imputed_confidence(Sprob,M,opts);
    consensusWeight=viewWeight;
    if isfield(opts,'consensusWeightMode') && ...
            strcmpi(opts.consensusWeightMode,'uniform')
        consensusWeight=ones(1,V);
    end
    H=update_consensus(Sview,M,consensusWeight,opts.imputedConsensusWeight, ...
        imputedConfidence,opts.lambda,opts.gamma,spatialD,H);

    % CIMLN-style intervention: randomly hide a subset of currently
    % observed views, form a counterfactual consensus, and enforce agreement.
    Pbar=fuse_views(Sprob,M,consensusWeight,opts.imputedConsensusWeight, ...
        imputedConfidence);
    Mcf=intervene_mask(M,opts.interventionProb,viewWeight);
    Pcf=fuse_views(Sprob,Mcf,consensusWeight,opts.imputedConsensusWeight, ...
        imputedConfidence);
    blend=min(max(opts.invariantBlend,0),1);
    Pbar=(1-blend)*Pbar+blend*Pcf;
    H=project_columns((1-blend)*H+blend*Pbar');
    Pbar=project_rows(Pbar);

    % Conservative soft cluster-frequency correction for long-tail classes.
    mass=sum(Pbar,1)+1e-10;
    cw=(mass/mean(mass)).^(-opts.classExponent);
    cw=min(max(cw,opts.classWeightMin),opts.classWeightMax);
    Q=bsxfun(@times,Pbar,cw);
    Q=bsxfun(@rdivide,Q,max(sum(Q,2),1e-10));
    b=min(max(opts.classBlend,0),1);
    H=project_columns((1-b)*H+b*Q');

    % TIC-IMV-inspired exact Shapley value over view subsets (V is normally 2/3)
    % and unbalanced OT discrepancy between each view and the consensus.
    contribution=ones(1,V); otBias=zeros(1,V);
    if opts.useContributionBalance
        contribution=shapley_contributions(Sprob,M,H);
    end
    if opts.useDistributionBalance
        for v=1:V
            idx=find(M(:,v));
            P=project_rows(Sprob{v}(:,idx)');
            Qot=project_rows(H(:,idx)');
            otBias(v)=unbalanced_ot_bias(P,Qot);
        end
    end
    progress=min(1,it/max(1,opts.maxIter*opts.curriculumWarmup));
    candidate=make_weights(obsRate,X,A,Z,M,contribution,otBias, ...
        progress,opts);
    momentum=min(max(opts.weightMomentum,0),0.99);
    viewWeight=momentum*viewWeight+(1-momentum)*candidate;
    viewWeight=bounded_weights(viewWeight,opts.minViewWeight,opts.maxViewWeight);
    newAlpha=osmagc_residual_weights(X,A,Z,M);
    if isfield(opts,'alphaMomentum')
        am=min(max(opts.alphaMomentum,0),0.99);
        alpha=am*alpha+(1-am)*newAlpha;
    else
        alpha=newAlpha;
    end

    change=norm(H(:)-prevH(:))/max(norm(prevH(:)),1e-10);
    graphResidual=0;
    for v=1:V
        idx=find(M(:,v));
        graphResidual=graphResidual+norm(X{v}(:,idx)-A{v}*Z{v}(:,idx),'fro')^2;
    end
    objectiveHistory(it)=surrogate_objective(X,A,Z,R,H,M,viewWeight,alpha, ...
        opts,spatialD,Pbar,Q,consensusWeight,imputedConfidence);
    history(it,:)=[change,graphResidual,mean(otBias),norm(updateNorm)];
    Hlist{it}=H;
    weightHistory(it,:)=viewWeight;
    contributionHistory(it,:)=contribution;
    otBiasHistory(it,:)=otBias;
    updateScaleHistory(it,:)=scales;
    alphaHistory(it,:)=alpha;
    prevH=H;
    if opts.verbose
        fprintf('TIC-IMV iter %d: change %.3e, weights [%s], OT [%s]\n', ...
            it,change,sprintf(' %.3f',viewWeight),sprintf(' %.3f',otBias));
    end
    if it>1 && change<opts.tol, break; end
end

history=history(1:it,:); weightHistory=weightHistory(1:it,:);
contributionHistory=contributionHistory(1:it,:);
otBiasHistory=otBiasHistory(1:it,:); updateScaleHistory=updateScaleHistory(1:it,:);
alphaHistory=alphaHistory(1:it,:);
objectiveHistory=objectiveHistory(1:it);
Hlist=Hlist(1:it);
labels=[];
if opts.returnLabels, [~,labels]=max(H,[],1); labels=labels(:); end
model=struct('A',{A},'Z',{Z},'R',{R},'H',H,'viewWeight',viewWeight, ...
    'obsRate',obsRate,'history',history,'viewWeightHistory',weightHistory, ...
    'viewContributionHistory',contributionHistory,'viewOTBiasHistory',otBiasHistory, ...
    'viewUpdateScaleHistory',updateScaleHistory,'alphaHistory',alphaHistory, ...
    'objectiveHistory',objectiveHistory, ...
    'labels',labels, ...
    'clusterNum',K,'mask',M,'options',opts,'Hlist',{Hlist},'tensorUsed',false);
end

function out=merge_options(base,user)
out=base; fn=fieldnames(user);
for i=1:numel(fn), out.(fn{i})=user.(fn{i}); end
end

function X=normalize_columns(X)
n=sqrt(sum(X.^2,1)); X=bsxfun(@rdivide,X,max(n,1e-12));
end

function A=init_anchors(X,m,opts)
if size(X,2)<=m
    A=X;
    if size(A,2)<m, A(:,end+1:m)=repmat(mean(X,2),1,m-size(A,2)); end
    return;
end
if exist('litekmeans','file')==2
    try
        lab=litekmeans(X',m,'replicates',opts.initReplicates,'maxiter',opts.initMaxIter);
        A=zeros(size(X,1),m);
        for j=1:m
            ix=lab(:)==j;
            if any(ix), A(:,j)=mean(X(:,ix),2);
            else, A(:,j)=X(:,randi(size(X,2))); end
        end
        return;
    catch
    end
end
perm=randperm(size(X,2),m); A=X(:,perm);
end

function A=init_anchors_from_labels(X,labels,m,opts)
labels=double(labels(:)); classes=unique(labels(isfinite(labels)));
A=zeros(size(X,1),m); filled=0;
for j=1:numel(classes)
    ix=labels==classes(j);
    if any(ix), filled=filled+1; A(:,filled)=mean(X(:,ix),2); end
    if filled>=m, break; end
end
if filled<m
    Afull=init_anchors(X,m,opts);
    if filled==0, A=Afull;
    else, A(:,filled+1:m)=Afull(:,filled+1:m); end
end
end

function Z=anchor_graph(X,A,k)
n=size(X,2); m=size(A,2); k=min(max(1,round(k)),m-1);
d=max(0,bsxfun(@plus,sum(X.^2,1)',sum(A.^2,1))-2*(X'*A));
[ds,ix]=sort(d,2,'ascend'); Z=zeros(m,n);
for i=1:n
    id=ix(i,1:k+1); di=ds(i,1:k+1);
    den=k*di(end)-sum(di(1:k));
    if den<=1e-12, w=zeros(k+1,1); w(1)=1;
    else, w=max((di(end)-di(1:k))/den,0); w=[w(:);0]; end
    if sum(w)<=1e-12, w(1)=1; end
    Z(id,i)=w/sum(w);
end
end

function Q=orthogonal_update(X,Z,Q0)
M=X*Z';
if isempty(M), Q=Q0; return; end
[u,~,v]=svd(M,'econ'); Q=u*v';
if any(~isfinite(Q(:))), Q=Q0; end
end

function H=init_consensus(Z,R,M,K,N)
H=zeros(K,N); den=zeros(1,N);
for v=1:numel(Z)
    q=double(M(:,v)'); H=H+(R{v}*Z{v}).*q; den=den+q;
end
H=bsxfun(@rdivide,H,max(den,1e-10));
for i=1:N, H(:,i)=simplex(H(:,i)); end
end

function H=init_osmagc_consensus(K,N)
% Match OSMAGC's deterministic orthogonal H initialization.
p=zeros(K,N); p(:,1:min(K,N))=eye(min(K,N));
[H,~,~]=svd(p','econ'); H=H';
if size(H,2)<N, H(:,end+1:N)=0; end
end

function alpha=osmagc_residual_weights(X,A,Z,M)
V=numel(X); energy=zeros(1,V);
for v=1:V
    idx=find(M(:,v));
    residual=X{v}(:,idx)-A{v}*Z{v}(:,idx);
    energy(v)=sum(residual(:).^2);
end
total=sum(energy);
if ~isfinite(total) || total<=1e-12
    alpha=ones(1,V)/V;
else
    alpha=energy/total;
    alpha=max(alpha,1e-8); alpha=alpha/sum(alpha);
end
end

function H=update_consensus(S,M,w,imputedWeight,confidence,lambda,gamma,D,H0)
N=size(M,1); K=size(S{1},1); V=numel(S);
weighted=zeros(K,N);
for v=1:V
    q=w(v)*(double(M(:,v)')+imputedWeight*double(~M(:,v)').*confidence(:,v)');
    weighted=weighted+S{v}.*q;
end
% Match OSMAGC's H block: for gamma=0 it projects the SUM of aligned view
% graphs onto the simplex.  Averaging first changes the Euclidean simplex
% projection and can severely alter cluster margins, especially for views
% with different feature scales or incomplete masks.
if gamma>0 && ~isempty(D)
    H=(lambda*weighted)/(gamma*D+lambda*speye(N));
else
    H=weighted;
end
H=project_columns(H);
if isempty(H), H=H0; end
end

function P=fuse_views(S,M,w,imputedWeight,confidence)
N=size(M,1); K=size(S{1},1); P=zeros(N,K); den=zeros(N,1);
for v=1:numel(S)
    q=w(v)*(double(M(:,v))+imputedWeight*double(~M(:,v)).*confidence(:,v));
    P=P+bsxfun(@times,S{v}',q); den=den+q;
end
P=bsxfun(@rdivide,P,max(den,1e-10)); P=project_rows(P);
end

function C=compute_imputed_confidence(S,M,opts)
% Unsupervised confidence for completed memberships.  Observed memberships
% retain unit weight; missing memberships receive an entropy-based weight
% with a floor so that no view disappears from the completion objective.
N=size(M,1); V=numel(S); C=ones(N,V);
if ~isfield(opts,'imputedConfidenceMode') || ...
        strcmpi(opts.imputedConfidenceMode,'none')
    return;
end
K=size(S{1},1); floorValue=min(max(opts.imputedConfidenceFloor,0),1);
den=max(log(max(K,2)),1e-12);
for v=1:V
    P=max(real(S{v}),0); P=bsxfun(@rdivide,P,max(sum(P,1),1e-12));
    ent=-sum(P.*log(max(P,1e-12)),1)/den;
    conf=floorValue+(1-floorValue)*(1-ent);
    C(:,v)=conf(:); C(M(:,v),v)=1;
end
end

function Mcf=intervene_mask(M,p,w)
[N,V]=size(M); Mcf=M;
drop=M & (rand(N,V)<min(max(p,0),1)); Mcf(drop)=false;
for i=1:N
    if any(M(i,:)) && ~any(Mcf(i,:))
        ix=find(M(i,:)); [~,j]=max(w(ix)); Mcf(i,ix(j))=true;
    end
end
end

function w=initial_weights(rate,opts)
ref=mean(rate(rate>0)); if isempty(ref), ref=1; end
w=((ref+1e-10)./(rate+1e-10)).^opts.missingExponent;
w=bounded_weights(w,opts.minViewWeight,opts.maxViewWeight);
end

function w=make_weights(rate,X,A,Z,M,phi,ot,progress,opts)
V=numel(X); eps0=1e-10; score=zeros(1,V); ref=mean(rate(rate>0));
if isempty(ref), ref=1; end
phi=positive_normalize(phi,opts.contributionTemperature);
% A contribution estimate is a noisy feedback signal, especially when a
% view has few observed samples.  Keep it as a small correction around one
% instead of allowing a low Shapley value to dominate the reconstruction.
phi=min(max(phi,0.75),1.25);
phi=phi/max(mean(phi),eps0);
for v=1:V
    ix=find(M(:,v));
    if isempty(ix), score(v)=eps0; continue; end
    res=X{v}(:,ix)-A{v}*Z{v}(:,ix);
    err=sum(res(:).^2)/max(numel(res),1);
    rel=(err+eps0)^(-opts.residualExponent);
    obs=((ref+eps0)/(rate(v)+eps0))^opts.missingExponent;
    if opts.useContributionBalance
        relContribution=phi(v)/max(phi);
        cf=(1/max(relContribution,0.75))^opts.contributionExponent;
        cf=min(max(cf,0.85),1.15);
        curriculum=1+progress*opts.curriculumStrength*max(1-relContribution,0);
    else
        cf=1; curriculum=1;
    end
    if opts.useDistributionBalance, df=exp(-opts.distributionExponent*max(ot(v),0));
    else, df=1; end
    score(v)=rel*obs*cf*curriculum*df;
end
if any(~isfinite(score)) || sum(score)<=eps0, w=initial_weights(rate,opts);
else, w=score/max(mean(score),eps0); end
w=bounded_weights(w,opts.minViewWeight,opts.maxViewWeight);
end

function w=bounded_weights(w,lo,hi)
w=max(double(w(:)'),1e-12); V=numel(w);
if lo>1 || hi<1 || lo>hi, error('View-weight bounds must satisfy lo <= 1 <= hi.'); end
w=w/max(mean(w),1e-12);
for t=1:20
    low=w<lo; high=w>hi; free=~(low|high);
    if ~any(low|high), break; end
    fixed=zeros(size(w)); fixed(low)=lo; fixed(high)=hi;
    if ~any(free), w=fixed; break; end
    remaining=V-sum(fixed);
    w(free)=w(free)/max(sum(w(free)),1e-12)*remaining;
    w(low)=lo; w(high)=hi;
end
end

function s=positive_normalize(x,temp)
x=real(x(:)'); x(~isfinite(x))=0; x=x-min(x);
if max(x)<=1e-10, x=ones(size(x)); else, x=x+1e-10; end
if temp>0 && temp~=1, x=x.^(1/temp); end
s=x/max(mean(x),1e-10);
end

function phi=shapley_contributions(S,M,H)
V=numel(S); N=size(M,1); K=size(H,1); U=zeros(1,2^V);
for mask=0:(2^V-1)
    use=false(1,V);
    for v=1:V, use(v)=bitget(mask,v)>0; end
    agg=zeros(K,N); cnt=zeros(1,N);
    for v=1:V
        if use(v)
            q=double(M(:,v)'); agg=agg+S{v}.*q; cnt=cnt+q;
        end
    end
    valid=cnt>0;
    if any(valid)
        agg=bsxfun(@rdivide,agg(:,valid),cnt(valid));
        delta=agg-H(:,valid);
        mse=sum(delta(:).^2)/max(K*sum(valid),1);
        U(mask+1)=mean(valid)*exp(-mse);
    end
end
phi=zeros(1,V); fact=factorial(V);
for v=1:V
    for mask=0:(2^V-1)
        if bitget(mask,v)==0
            s=sum(bitget(mask,1:V)); coeff=factorial(s)*factorial(V-s-1)/fact;
            phi(v)=phi(v)+coeff*(U(mask+2^(v-1)+1)-U(mask+1));
        end
    end
end
% Preserve a positive floor.  Subtracting the minimum and then using the
% result as an inverse weight turns the weakest view into an artificial
% dominant view whenever its estimated contribution is near zero.
phi=real(phi(:)'); phi(~isfinite(phi))=0;
center=max(mean(phi),1e-10);
phi=phi+0.25*center;
if max(abs(phi))<=1e-10, phi=ones(1,V); end
end

function b=unbalanced_ot_bias(P,Q)
% Generalized Sinkhorn divergence for cluster marginals and membership
% geometry. Inputs are n-by-K simplex rows for the same observed samples.
eps0=1e-8; n=size(P,1); K=size(P,2);
if n<1, b=0; return; end
a=mean(P,1)'+eps0; a=a/sum(a); c=mean(Q,1)'+eps0; c=c/sum(c);
pn=bsxfun(@rdivide,P,max(sqrt(sum(P.^2,1)),eps0));
qn=bsxfun(@rdivide,Q,max(sqrt(sum(Q.^2,1)),eps0));
sim=max(0,min(1,pn'*qn));
C=1-sim; epsilon=0.08; rho=0.5; theta=rho/(rho+epsilon);
G=exp(-C/epsilon)+eps0; u=ones(K,1); v=ones(K,1);
for t=1:40
    u=(a./max(G*v,eps0)).^theta;
    v=(c./max(G'*u,eps0)).^theta;
end
T=bsxfun(@times,bsxfun(@times,G,u),v'); mass=sum(T(:));
if mass<=eps0, b=1; else, b=sum(sum(T.*C))/mass; end
b=max(0,min(1,b));
end

function X=project_columns(X)
for i=1:size(X,2), X(:,i)=simplex(X(:,i)); end
end
function X=project_rows(X)
for i=1:size(X,1), X(i,:)=simplex(X(i,:)')'; end
end
function x=simplex(x)
x=double(real(x(:))); if isempty(x), return; end
u=sort(x,'descend'); cssv=cumsum(u)-1;
rho=find(u-cssv./(1:numel(u))'>0,1,'last');
if isempty(rho), x(:)=1/numel(x); return; end
theta=cssv(rho)/rho; x=max(x-theta,0); s=sum(x);
if s<=1e-12, x(:)=1/numel(x); else, x=x/s; end
end

function D=grid_laplacian(sz)
H=sz(1); W=sz(2); N=H*W; grid=reshape(1:N,H,W);
a1=grid(1:max(H-1,0),:); b1=grid(2:H,:);
a2=grid(:,1:max(W-1,0)); b2=grid(:,2:W);
a=[a1(:);a2(:)]; b=[b1(:);b2(:)]; E=numel(a);
ii=[a;b;a;b]; jj=[a;b;b;a]; vv=[ones(E,1);ones(E,1);-ones(E,1);-ones(E,1)];
D=sparse(ii,jj,vv,N,N);
end

function value=surrogate_objective(X,A,Z,R,H,M,w,alpha,opts,D,Pbar,Q,consensusWeight,confidence)
%SURROGATE_OBJECTIVE Label-free iteration-wise surrogate objective snapshot.
% This is the implementation-level surrogate corresponding to Eq. (34).
% Feedback weights and intervention/class targets are recomputed at every
% iteration, so the complete stochastic solver is not claimed to descend one
% fixed differentiable objective at every step.
V=numel(X); N=size(M,1);
rec=0; con=0;
for v=1:V
    idx=find(M(:,v));
    if isempty(idx), continue; end
    % Effective reconstruction weight used by the implemented Z update.
    rw=(1+max(opts.zImbalanceScale,0)*(w(v)-1))/max(alpha(v),1e-8);
    e=X{v}(:,idx)-A{v}*Z{v}(:,idx);
    rec=rec+rw*sum(e(:).^2);
    d=R{v}*Z{v}-H;
    q=double(M(:,v)');
    if ~isempty(consensusWeight)
        q=consensusWeight(v)*q;
        if ~isempty(confidence)
            q=q+consensusWeight(v)*opts.imputedConsensusWeight* ...
                double(~M(:,v)').*double(confidence(:,v)');
        end
    end
    con=con+sum(sum(bsxfun(@times,d.^2,q)));
end
spa=0;
if ~isempty(D)
    spa=real(sum(sum((H*D).*H)));
end
cf=0;
if ~isempty(Pbar) && isequal(size(Pbar'),size(H))
    betaCf=max(opts.invariantBlend,0)/max(1-max(opts.invariantBlend,0),1e-12);
    d=H-Pbar'; cf=betaCf*sum(d(:).^2);
end
cls=0;
if ~isempty(Q) && isequal(size(Q'),size(H))
    betaCls=max(opts.classBlend,0)/max(1-max(opts.classBlend,0),1e-12);
    d=H-Q'; cls=betaCls*sum(d(:).^2);
end
value=rec+opts.lambda*con+opts.gamma*spa+cf+cls;
if ~isfinite(value), value=realmax('double')/1e6; end
end


