function [choiceProbs trialrec memRec taskvars opts] = choicemodel_ctxSample_2LR(subjData, varargin)
% function [choiceProbs trialrec memRec taskvars opts] = choicemodel_ctxSample_2LR(subjData, ...)
%
% Run (or simulate) the context-aware episode sampling model, at the given parameters.
%
% INPUTS:
%   subjData - Subject data. If empty, simulate and return a populated trialrec.
%              If not empty, return the probability that the model would have made each choice.
%
% OPTIONS:
%   verbose     (false) General debugging
%   veryverbose (false) Trial-by-trial updates
%
%   [Model parameters: These take values across a range and can be fit]
%   subjAlpha    (0.4)  (0,1)
%   subjAlphaMem (0.4)  (0,1)
%   subjBeta     (1)    Unconstrained
%   subjCtxAC    (0.95) (0,1)
%   subjPersev   (0)    Unconstrained
%   subjSamples  (6)    Positive integers
%
%   [Model flags: These qualitatively change the behavior of the model, and should be treated as distinct variants]
%   decayType   (0; 0 = Single forgetting function across trials, 1 = Separate function for trials of each option.)
%   whichModel  (0; 0 = Sampler, 1 = TD, 2 = Hybrid)
%   mcSamples   (500; When fitting, re-run each choice this many times, and compute choice probabilities as a fraction of those outcomes)
%   accumulateSamples (false; Run an accumulation to bound, rather than a fixed number of samples.) XXX unimp
%
%   [Empirical characteristics of the subject to be simulated]
%   memAccuracy  (0.95; What percentage of memory trials were correct. This is the average across subjects in my ctxbandit experiments.)
%   memConfident (0.85; What percentage of correct memory responses were high confidence. Average from ctxbandit experiments.)
%
%   [Simulated task structure] (I attempted to generalize this a bit, but it'd still need some work to generalize.)
%   numTrials       (360)
%   numBandits      (3)
%   numProbeTrials  (60)
%   fracValidProbes (50/60)
%   roomLen         (30)
%   numRooms        (6)
%   payoffSwitch    (10)
%
%
% RETURNS:
%   choiceProbs Probability subject would have made each choice. If simulation, these are all NaN.
%   trialrec    Simulated trialrec, or as passed.
%   memRec      Simulated memRec, or as passed.
%   taskvars    Simulated taskvars, if applicable.
%   opts        Options as set.
%

%%
% TODO:
%   - Make faster.
%       - Make the core choice simulation more efficient.
%   - Implement hybrid model-free and episode choices.
%       - Different alphas for direct experience and simulated experience.
%       - Allow simulated experience to update cached values.
%   - Try adaptive sampling to threshold, rather than a fixed number of samples. (There are arguments for either approach.)
%   - Experiment with temporal weighting within contexts.
%   - Other similarity/priority kernels besides time and explicit context; e.g. RPE, recent reward history.
%   - Perseveration as a smooth kernel, rather than one-back.
%

%%%%
% Sorry, this is all much more byzantine than it probably needs to be.
% Someday I may simplify, but for now I wanted to make sure I copied the task code in as directly as possible, for simulations.
%%%%

%%
% Set constants and parse options.

% taskvars contains task variables to be passed to (and potentially modified by) choice and memory probe functions.
taskvars.MODEL_SAMPLER  = 0;
taskvars.MODEL_TD       = 1;  % XXX unimp
taskvars.MODEL_HYBRID   = 2;  % XXX unimp
taskvars.DECAYTYPE_COMBINED = 0;        % Decay values by trial; as a choice episode gets older, it's less likely to be sampled.
taskvars.DECAYTYPE_BYOPTION = 1;        % Decay values by choice of each option; this yields a functional form more precisely matched to TD, but could arguably be relevant to non-cued episode sampling. (To be continued...)
taskvars.responseButtons    = [1 2 3 4];

% Task structure.
taskvars.optdefaults.numTrials       = 360;
taskvars.optdefaults.numBandits      = 3;
taskvars.optdefaults.numProbeTrials  = 60;
taskvars.optdefaults.fracValidProbes = 50/60;
taskvars.optdefaults.roomLen         = 30;
taskvars.optdefaults.numRooms        = 6;
taskvars.optdefaults.payoffSwitch    = 10;

% Subject characteristics.
taskvars.optdefaults.memAccuracy  = 0.95;    % dist?
taskvars.optdefaults.memConfident = 0.85;    % dist?
taskvars.optdefaults.subjAlpha    = 0.4;     % \alpha = learning rate / decay rate                         % dist? set to fit values?
taskvars.optdefaults.subjAlphaMem = 0.2;     % \alpha_{mem} = learning / decay rate on reinstatements      % dist? set to fit values?
taskvars.optdefaults.subjBeta     = 1;       % \beta  = softmax temp                                       % dist? set to fit values?
taskvars.optdefaults.subjCtxAC    = 0.95;    % \pi    = context autocorrelation between successive samples % dist? set to fit values?
taskvars.optdefaults.subjPersev   = 0;       % Choice stickiness
taskvars.optdefaults.subjSamples  = 6;       % # of samples to draw for each choice                        % dist? set to fit values? 0 == use adaptive threshold (XXX unimp)?

taskvars.optdefaults.accumulateSamples = false;

taskvars.optdefaults.mcSamples    = 500;
taskvars.optdefaults.whichModel   = taskvars.MODEL_SAMPLER;
taskvars.optdefaults.decayType    = taskvars.DECAYTYPE_COMBINED;

[opts] = parse_args(varargin, ...
                              'verbose', false, ...
                              'veryverbose', false, ...
                              'numTrials', taskvars.optdefaults.numTrials, ...
                              'numBandits', taskvars.optdefaults.numBandits, ...
                              'numProbeTrials', taskvars.optdefaults.numProbeTrials, ...
                              'fracValidProbes', taskvars.optdefaults.fracValidProbes, ...
                              'roomLen', taskvars.optdefaults.roomLen, ...
                              'numRooms', taskvars.optdefaults.numRooms, ...
                              'payoffSwitch', taskvars.optdefaults.payoffSwitch, ...
                              'memAccuracy', taskvars.optdefaults.memAccuracy, ...
                              'memConfident', taskvars.optdefaults.memConfident, ...
                              'subjAlpha', taskvars.optdefaults.subjAlpha, ...
                              'subjAlphaMem', taskvars.optdefaults.subjAlphaMem, ...
                              'subjBeta', taskvars.optdefaults.subjBeta, ....
                              'subjPersev', taskvars.optdefaults.subjPersev, ...
                              'subjCtxAC', taskvars.optdefaults.subjCtxAC, ...
                              'subjSamples', taskvars.optdefaults.subjSamples, ...
                              'accumulateSamples', taskvars.optdefaults.accumulateSamples, ...
                              'mcSamples', [], ...
                              'whichModel', taskvars.optdefaults.whichModel, ...
                              'decayType', taskvars.optdefaults.decayType);
if (nargin < 1)
    subjData = [];
end

if (isempty(subjData))
    opts.simulateSubj = true;
else
    opts.simulateSubj = false;
end

% We allow the user to override mcSamples when simulating. For instance, if we want to simulate subjects with more than one sample.
if (isempty(opts.mcSamples))
    if (opts.simulateSubj)
        % Simulating: Just one draw.
        opts.mcSamples = 1;
    else
        % Fitting: Approximate distribution using mcSamples # of samples
        opts.mcSamples = taskvars.optdefaults.mcSamples;
    end
end

% Discretize
opts.subjSamples = ceil(opts.subjSamples);

%%
% Initialize internal state.
taskvars.episodeList       = cell(opts.numTrials+1, 1);
taskvars.banditEpisodeList = cell(opts.numBandits, opts.numTrials+1);
taskvars.numBanditEpisodes = zeros(1, 3);

taskvars.trialIdx = 0;

choiceProbs = ones(opts.numTrials, 1)*(1/opts.numBandits);

if (~isempty(subjData))
    %% Fit subjData
    trialrec = subjData.trialrec;

    try
        memRec = subjData.memRec;
    catch
        memRec = [];
    end
    taskvars.choiceBlocks       = subjData.choiceBlocks;
    taskvars.invalidProbeTrials = sort(subjData.invalidProbeTrials);
    taskvars.memProbeTrials     = setdiff(sort(subjData.memProbeTrials), taskvars.invalidProbeTrials);
    taskvars.contexts           = [0:opts.numRooms];
else
    %% Red pill!

    %%% Set up task structures.
    % TODO: To generalize for other tasks, these could be subclassed out to return taskvars and taskfuncs.

    % Initialize recording structures.
    trialrec = cell(opts.numTrials, 1);
    memRec   = [];

    % Payout characteristics.
    taskvars.initPayouts  = [60 30 10];
    taskvars.initPayouts  = taskvars.initPayouts(randperm(length(taskvars.initPayouts)));
    taskvars.decayTheta   = taskvars.initPayouts;
    taskvars.decayLambda  = 0.6;
    taskvars.driftSigma   = 8;
    taskvars.driftNoise   = chol(eye(1)*(taskvars.driftSigma^2));
    taskvars.payoffBounds = [5 95];
    taskvars.ctxBump      = 3;

    % Generate sequence of choice and memory probe trials for final room.
    numProbes = opts.payoffSwitch * (opts.numRooms);
    meanCT    = 5;
    maxCT     = 8;
    minCT     = 2;

    choiceBlocks                     = -ceil(log(rand(1, numProbes))./(1/meanCT))+minCT;
    choiceBlocks(choiceBlocks>maxCT) = maxCT; % Trim blocks greater than maxCT

    % Trim the generated choice blocks until:
    %   1. They sum to the length of the final room (opts.numTrials/2)
    %   2. They fit within (minCT, maxCT)
    while (((sum(choiceBlocks)) ~= (opts.numTrials/2)) || ...
             any(choiceBlocks>maxCT) || ...
             any(choiceBlocks<minCT))
        ind               = ceil(rand*(numProbes));      % Pick a random block to trim
        choiceBlocks(ind) = choiceBlocks(ind) - sign(sum(choiceBlocks) - (opts.numTrials/2));

        choiceBlocks(choiceBlocks<minCT) = minCT;
        choiceBlocks(choiceBlocks>maxCT) = maxCT;
    end

    choiceBlocks = choiceBlocks - 1; % Leaves room for numProbes
    taskvars.choiceBlocks   = choiceBlocks;

    % Place a memory probe trial at the end of each choice block.
    taskvars.memProbeTrials = cumsum(taskvars.choiceBlocks+1)+[opts.roomLen*opts.numRooms];

    if (opts.verbose)
        disp(['choicemodel_ctxSample: Generated choice trials lengths,' ...
              ' sum '  num2str(sum(taskvars.choiceBlocks'))  ...
              ' mean ' num2str(mean(taskvars.choiceBlocks'))]);
        taskvars.choiceBlocks
        taskvars.memProbeTrials
    end

    trialNums = 1:opts.numTrials;
    taskvars.choiceTrials = setdiff(trialNums, taskvars.memProbeTrials);

    % Shuffle the list of memory probe trials.
    taskvars.memProbeTrials = taskvars.memProbeTrials(randperm(length(taskvars.memProbeTrials)));

    % Now take the first numInvalidProbes # of indexes. These will be the invalid/lure probes (novel images).
    taskvars.numInvalidProbes   = ceil((1-opts.fracValidProbes) * opts.numProbeTrials);
    taskvars.invalidProbeTrials = sort(taskvars.memProbeTrials(1:taskvars.numInvalidProbes));

    % Now the first opts.payoffSwitch trials of each context are available for later memory probes.
    taskvars.availableForMemProbe = [];
    for bIdx = 1:opts.numRooms-1;
        taskvars.availableForMemProbe = [taskvars.availableForMemProbe (opts.roomLen*bIdx):((opts.roomLen*bIdx)+opts.payoffSwitch-1)];
    end

    % Initialize contexts and probe images
    taskvars.contexts = zeros(1,opts.numTrials);
    % First context is short
    taskvars.contexts(1:(opts.roomLen-opts.payoffSwitch)) = 0;
    for ci = 1:opts.numRooms-1;
        sp = (opts.roomLen - opts.payoffSwitch) + ((ci-1)*opts.roomLen);
        ep = sp + opts.roomLen;
        taskvars.contexts((sp+1):ep) = ci;
    end

    taskvars.contexts(:, ((ep+1):end)) = max(taskvars.contexts)+1;

    taskvars.payout       = zeros(opts.numBandits, opts.numTrials);
    taskvars.payout(:, 1) = taskvars.initPayouts;
end % if (~isempty(subjData))

% If not simulating, and don't have a memRec, reconstruct memRec here.
% (First few subjects had an incomplete memRec; reconstructing here makes fitting easier.)
if (isempty(memRec) && ~opts.simulateSubj)
    %%
    % Generate memRec structure as in doPostTest
    memPairs          = {};
    for mpi = 1:length(taskvars.memProbeTrials)
        probed = trialrec{taskvars.memProbeTrials(mpi)}.probed;

        ft = [];
        for thisIdx = 1:(taskvars.memProbeTrials(mpi)-1);
            if (probed == trialrec{thisIdx}.probed)
                ft(1) = thisIdx;
                break;
            end
        end

        if (~isempty(ft))
            memPairs{end+1} = [taskvars.memProbeTrials(mpi) ft(1)];
        end
    end

    % Test each mem probe trial
    for testIdx = 1:length(memPairs)
        % Index of current trial, index of evoked bandit, response on room test, correct (1)/incorrect (-1) response on room test
        memRec(testIdx, :) = [memPairs{testIdx}(1) memPairs{testIdx}(2)];
    end
end % if(isempty(memRec))

%%
% First half of trials - `context rooms'.
for bIdx = 1:opts.numRooms;
    if (opts.verbose)
        disp(['choicemodel_ctxSample: Entering room ' num2str(bIdx)]);
    end

    for t = 1:opts.roomLen;
        taskvars.trialIdx   = taskvars.trialIdx + 1;
        [choiceProbs(taskvars.trialIdx) trialrec taskvars] = doChoiceTrial(trialrec, taskvars, opts);
    end
end

%%
% Second half of trials - probe session / `empty room'.
for cb = 1:size(taskvars.choiceBlocks, 2);
    % Run the prescribed # of choice trials
    for ct = 1:taskvars.choiceBlocks(cb);
        taskvars.trialIdx   = taskvars.trialIdx + 1;
        [choiceProbs(taskvars.trialIdx) trialrec taskvars] = doChoiceTrial(trialrec, taskvars, opts);
    end

    % Do a memory probe
    taskvars.trialIdx   = taskvars.trialIdx + 1;
    [trialrec taskvars] = doMemProbe(trialrec, memRec, taskvars, opts);
end % for choiceBlocks

end % function choicemodel_ctxSample

%%
% Run a choice trial.
%
% XXX: A great many inefficiencies here. Tradeoffs were made in the interest of time and readability. Moore's law, save us.
%
function [cp trialrec taskvars] = doChoiceTrial(trialrec, taskvars, opts)
    % Probability of choosing each bandit on this trial.
    cp = zeros(opts.numBandits, 1);

    if (opts.simulateSubj)
        % Simulation.

        % Generate payoffs for this trial.
        if (mod(taskvars.trialIdx, opts.roomLen) == 0)
            bestOpt = find(taskvars.decayTheta==max(taskvars.decayTheta));

            while(taskvars.decayTheta(bestOpt) == max(taskvars.decayTheta))
                taskvars.decayTheta = taskvars.decayTheta(randperm(length(taskvars.decayTheta)));
            end
        end

        if (taskvars.trialIdx > 1)
            if (mod(taskvars.trialIdx, opts.roomLen) < taskvars.ctxBump)
                % Slow transitions to emphasize payoff structure
                decayLambda_eff = 0.95;
            else
                decayLambda_eff = taskvars.decayLambda;
            end

            for thisBandit = 1:opts.numBandits;
                % Drift bandits
                %   \mu_{i,t} = \lambda * \mu_{i,t-1} + (1 - \lambda) * \theta + \nu
                taskvars.payout(thisBandit, taskvars.trialIdx) = decayLambda_eff*taskvars.payout(thisBandit, taskvars.trialIdx-1) + ...
                                                                 (1-decayLambda_eff)*taskvars.decayTheta(thisBandit) + ...
                                                                 randn(1)*taskvars.driftNoise;

                % Reflect at specified boundaries [lower, upper]
                if (taskvars.payout(thisBandit, taskvars.trialIdx) > taskvars.payoffBounds(2))
                    taskvars.payout(thisBandit, taskvars.trialIdx) = taskvars.payoffBounds(2) - ...
                                                                      (taskvars.payout(thisBandit, taskvars.trialIdx) - taskvars.payoffBounds(2));
                end
                if (taskvars.payout(thisBandit, taskvars.trialIdx) < taskvars.payoffBounds(1))
                    taskvars.payout(thisBandit, taskvars.trialIdx) = taskvars.payoffBounds(1) + ...
                                                                      (taskvars.payoffBounds(1) - taskvars.payout(thisBandit, taskvars.trialIdx));
                end
            end
        end

        trialrec{taskvars.trialIdx}.bandits    = taskvars.payout(:,taskvars.trialIdx);
        trialrec{taskvars.trialIdx}.decayTheta = taskvars.decayTheta;
        trialrec{taskvars.trialIdx}.contexts   = taskvars.contexts(taskvars.trialIdx);
    end % if (opts.simulateSubj


    %% Make choice.

    % If fitting, run this opts.mcSamples times, then set cp to the fraction of times model chose the option the subject actually chose.
    % If simulating, run once (unless intentionally simulating mcSamples>1).

    % Draw samples.
    if (taskvars.trialIdx == 1)
        % First trial - just pick at random.
        chosenBandit = ceil(rand(1)*opts.numBandits);
        cp = ones(opts.numBandits, 1)*(1/opts.numBandits);
    else
        for mcIdx = 1:opts.mcSamples;
            if (opts.whichModel == taskvars.MODEL_SAMPLER)
                % Set sampleContext to -1 at first, indicating samples are drawn according to time rather than explicit context.
                % XXX: Ideally this would use a similarity function that takes all of these things into account,
                %      perhaps motivated at the implementation level by a spreading activation or accumulator model. To be continued...
                sampleContext    = -1;

                sampleChoice = [];
                sampleValue  = [];

                for sampleIdx = 1:opts.subjSamples;
                    gotSample = false;
                    while (~gotSample)
                        % NB: opts.subjCtxAC is essentially a way to break out of the sample context.
                        if (sampleContext ~= -1 && rand(1)<opts.subjCtxAC && ...
                            (sampleContext ~= max(taskvars.contexts) || ...
                             (sampleContext ~= max(taskvars.contexts) && sampleContext ~= trialrec{taskvars.trialIdx-1}.contexts)))
                            % If a sample context is set - and we're not in one of the cases where a sample context doesn't make sense - then draw from the sample context
                            %    with uniform probability (gives approximate power-law forgetting function over trials, on average, and possibly a hyperbolic discounting when doing prospective/constructive sampling; To be continued...).

                            % NB: This is where to plug in models that give rise to nonuniform sampling probability within each context, influenced by salience/RPE/reward, primacy/recency. To be continued...
                            if (opts.decayType == taskvars.DECAYTYPE_COMBINED)
                                % XXXFASTER: Replace this find with a static reference (array of trial #s per context)
                                ctxtrials    = find(cellfun(@(x)(x.contexts==sampleContext), {trialrec{1:taskvars.trialIdx-1}}));
                                sampledTrial = ceil(rand(1)*length(ctxtrials));

                                sampledTrial = ctxtrials(sampledTrial);
                                if (~isempty(taskvars.episodeList{sampledTrial}))
                                    gotSample = true;
                                    sampleChoice(sampleIdx) = trialrec{sampledTrial}.choice;
                                    sampleValue(sampleIdx)  = sign(trialrec{sampledTrial}.rwdval)*2-1;
                                end
                            elseif (opts.decayType == taskvars.DECAYTYPE_BYOPTION)
                                gotSample = true;

                                for banditIdx = 1:opts.numBandits;
                                    if (~taskvars.numBanditEpisodes(banditIdx))
                                        continue;
                                    end
                                    % XXXFASTER: Replace this find with a static reference (array of trial #s per context, with chosen bandits)
                                    ctxtrials    = find(cellfun(@(x)(x.contexts==sampleContext && x.choice==banditIdx), {trialrec{1:taskvars.trialIdx-1}}));
                                    if (isempty(ctxtrials))
                                        % No samples for this bandit in this context. Skip it.
                                        continue;
                                    end
                                    sampledTrial = ceil(rand(1)*length(ctxtrials));
                                    sampledTrial = ctxtrials(sampledTrial);

                                    sampleChoice(end+1) = banditIdx;
                                    sampleValue(end+1)  = sign(trialrec{sampledTrial}.rwdval)*2-1;
                                end
                            end % if (opts.decayType ...
                        else    % No sample context set; Use temporal weighting
                            if (opts.decayType == taskvars.DECAYTYPE_COMBINED)
                                % Draw one sample.
                                sampleTrialProbs = opts.subjAlpha * (1-opts.subjAlpha).^[0:taskvars.trialIdx-1];
                                sampleTrialProbs = sampleTrialProbs./sum(sampleTrialProbs);
                                % XXXFASTER: How to replace this find? Not sure.
                                sampledTrial     = taskvars.trialIdx - (find(rand(1)<cumsum(sampleTrialProbs), 1, 'first')-1);

                                if (~isempty(sampledTrial) && ~isempty(taskvars.episodeList{sampledTrial}))
                                    gotSample = true;
                                    sampleContext = taskvars.episodeList{sampledTrial}.contexts;
                                    sampleChoice(sampleIdx) = taskvars.episodeList{sampledTrial}.choice;
                                    sampleValue(sampleIdx)  = sign(taskvars.episodeList{sampledTrial}.rwdval)*2-1;
                                end
                            elseif (opts.decayType == taskvars.DECAYTYPE_BYOPTION)
                                % Draw one sample for each bandit.
                                % Don't need to worry about gotSample here, because all of the banditEpisodeList caches only include valid episodes.

                                % XXX: How to set sampleContext???
                                %   I guess the `right' thing to do here is to set it for each one, in order. But then order is an issue.
                                %   I *think* that, functionally, this should never matter.
                                gotSample = true;
                                sampleIndices = [];
                                sampleContextCandidates = [];

                                for banditIdx = 1:opts.numBandits;
                                    if (~taskvars.numBanditEpisodes(banditIdx))
                                        continue;
                                    end
                                    sampleTrialProbs = opts.subjAlpha * (1-opts.subjAlpha).^[0:taskvars.numBanditEpisodes(banditIdx)-1];
                                    sampleTrialProbs = sampleTrialProbs./sum(sampleTrialProbs);
                                    % XXXFASTER: How to replace this find? Not sure.
                                    sampledTrial     = taskvars.numBanditEpisodes(banditIdx) - (find(rand(1)<cumsum(sampleTrialProbs), 1, 'first')-1);

                                    sampleChoice(end+1)  = banditIdx;
                                    sampleValue(end+1)   = sign(taskvars.banditEpisodeList{banditIdx,sampledTrial}.rwdval)*2-1;
                                    sampleIndices(end+1) = length(sampleChoice);
                                    sampleContextCandidates(end+1) = taskvars.banditEpisodeList{banditIdx,sampledTrial}.contexts;
                                end

                                % If any sample context differs from the current, switch to that.
                                % XXX: This is weird and I don't like it, but not really sure what to do. Again, functionally shouldn't matter for this application.
                                diffctx = find(sampleContextCandidates~=sampleContext & sampleContextCandidates~=-1);
                                if (~isempty(diffctx))
                                    sampleContextCandidates = sampleContextCandidates(diffctx);
                                    sampleContext = sampleContextCandidates(ceil(rand(1)*length(diffctx)));
                                    if (opts.veryverbose)
                                        disp(['choicemodel_ctxSample: Trial ' num2str(taskvars.trialIdx) ...
                                                ', MCsample ' num2str(mcIdx) ...
                                                ', switched to context: ' num2str(sampleContext)]);
                                    end
                                end

                                if (opts.veryverbose)
                                    disp(['choicemodel_ctxSample: Trial ' num2str(taskvars.trialIdx) ...
                                              ', MCsample ' num2str(mcIdx) ...
                                              ', sample ' num2str(sampleIdx) ...
                                              ', bandit ' num2str(sampleChoice(sampleIndices)) ...
                                              ', reward ' num2str(sampleValue(sampleIndices)) ...
                                              ', new sample context: ' num2str(sampleContext)]);
                                end
                            end % if (opts.decayType...
                        end % if (~sampleContext ...
                    end % while (~gotSample

                    if (opts.veryverbose)
                        if (opts.decayType == taskvars.DECAYTYPE_COMBINED)
                            disp(['choicemodel_ctxSample: Trial ' num2str(taskvars.trialIdx) ...
                                  ', MCsample ' num2str(mcIdx) ...
                                  ', sample ' num2str(sampleIdx) ...
                                  ': Trial ' num2str(trialrec{sampledTrial}.probed) ...
                                  ', bandit ' num2str(sampleChoice(sampleIdx)) ...
                                  ', reward ' num2str(sampleValue(sampleIdx)) ...
                                  ', new sample context: ' num2str(sampleContext)]);
                        end
                    end
                end % for sampleIdx

                % XXX: Could test shared value cache here; with subclassed other updates (e.g. Q-values).

                % XXXFASTER: This doesn't need to be a for loop, but the matrix mult was nigh-inscrutable.
                banditValue = NaN(opts.numBandits, 1);
                for banditIdx = 1:opts.numBandits;
                    if (opts.accumulateSamples)
                        % Rather than mean and exp, run an accumulator. XXX unimp, this is just a stub
                        banditValue(banditIdx) = sum(sampleValue(sampleChoice==banditIdx)) - sum(0.5*sampleValue(sampleChoice~=banditIdx));
                    else
                        banditValue(banditIdx) = mean(sampleValue(sampleChoice==banditIdx));
                    end
                end

                banditValue(isnan(banditValue)) = 0;

                lastChoiceTrial = taskvars.trialIdx-1;
                if (isempty(trialrec{lastChoiceTrial}.bandits))
                    lastChoiceTrial = taskvars.trialIdx-2;
                end
                persev = ((trialrec{lastChoiceTrial}.choice == [1:opts.numBandits]).*2)-1;
                persev = opts.subjPersev * persev';

                if (~opts.simulateSubj)
                    chosenBandit = trialrec{taskvars.trialIdx}.choice;
                    if (chosenBandit == 0)
                        persev = zeros(opts.numBandits,1);
                    end
                end

                % Compute choice probabilities.
                if (opts.accumulateSamples)
                    % Binary per sample. Multiple iterations will turn this into probabilities.
                    % XXX: Could compute distribution of threshold crossings, but that would require a 3-way race model for ctxSample.
                    banditProbs = banditValue==max(banditValue);
                else
                    denom       = exp(persev(1) + opts.subjBeta*banditValue(1)) + ...
                                  exp(persev(2) + opts.subjBeta*banditValue(2)) + ...
                                  exp(persev(3) + opts.subjBeta*banditValue(3));

                    banditProbs  = exp(persev + opts.subjBeta*banditValue)./denom;
                end
                cp = cp + banditProbs;

                if (opts.veryverbose)
                    disp(['choicemodel_ctxSample: Trial ' num2str(taskvars.trialIdx) ...
                          ', MCsample ' num2str(mcIdx) ...
                          ', sample values ' num2str(sampleValue.*sampleChoice) ...
                          ', bandit values ' num2str(banditValue') ...
                          ', bandit probabilities ' num2str(banditProbs')]);
                end

            end % if (whichModel==
        end % for mcIdx
    end % if (trialIdx == 1)

    if (~opts.simulateSubj)
        % Fitting. Get actual choice.
        chosenBandit = trialrec{taskvars.trialIdx}.choice;
    end

    % Get average choice probability for the chosen bandit.
    if (~opts.simulateSubj && chosenBandit == 0)
        % XXXFASTER: If not simulating, and this is a skipped trial, could have just skipped all the above.
        %            Few enough that it's not worth the extra indent.
        cp = 1/opts.numBandits;
    else
        cp = cp/opts.mcSamples;
    end

    if (opts.simulateSubj)
        % Simulation. Record choice.
        chosenBandit = find(rand(1)<cumsum(cp), 1, 'first');
        trialrec{taskvars.trialIdx}.choice = chosenBandit;
        % TODO: In the future we can simulate RT from at least the sampler models, assuming cue is coincident with decision start &/or we have enough early responses to make an MS-DDM applicable. To be continued...
        trialrec{taskvars.trialIdx}.RT     = -1;

        % Generate payout.
        if ((taskvars.trialIdx > taskvars.ctxBump) && (trialrec{taskvars.trialIdx}.contexts ~= trialrec{taskvars.trialIdx-taskvars.ctxBump}.contexts) && ...
            (trialrec{taskvars.trialIdx-taskvars.ctxBump}.contexts > -1))
            % If we're in the first few trials of a context, make the payout deterministic.
            % NB: This is meant to entice subjects, kept here to keep statistics matched.
            favopt = find(taskvars.decayTheta == max(taskvars.decayTheta));
            trialrec{taskvars.trialIdx}.bandits(favopt)    = 100;
            trialrec{taskvars.trialIdx}.decayTheta(favopt) = 100;
            isrwded = chosenBandit == favopt;
        else
            isrwded = rand(1)<(taskvars.payout(chosenBandit, taskvars.trialIdx)/100);
        end

        trialrec{taskvars.trialIdx}.rwdval = isrwded*10;
        trialrec{taskvars.trialIdx}.probed = taskvars.trialIdx;
    end % if (~opts.simulateSubj)

    % Save this trial in episodic memory.
    % TODO: Decay, along similarity dimensions. Don't like bitflips, see Foster for ideas about principled ways to do this.
    if (chosenBandit ~= 0)
        cp = cp(chosenBandit);

        taskvars.numBanditEpisodes(chosenBandit) = taskvars.numBanditEpisodes(chosenBandit) + 1;
        taskvars.banditEpisodeList{chosenBandit,taskvars.numBanditEpisodes(chosenBandit)} = trialrec{taskvars.trialIdx};
        taskvars.episodeList{taskvars.trialIdx} = trialrec{taskvars.trialIdx};
    else
        cp = 1/opts.numBandits;
    end

end % function doChoiceTrial

function [trialrec, taskvars] = doMemProbe(trialrec, memRec, taskvars, opts)
    taskvars.episodeList{taskvars.trialIdx} = [];

    % Answer with correctness & confidence as specified by opts.
    answerCorrect   = rand(1)<opts.memAccuracy;
    answerConfident = rand(1)<opts.memConfident;

    % Valid probe?
    if (~ismember(taskvars.trialIdx, sort(taskvars.invalidProbeTrials)))
        if (opts.simulateSubj)
            % Simulating. Generate a response.
            response = (answerCorrect & answerConfident) * 1 + ...
                       (answerCorrect & ~answerConfident) * 2 + ...
                       (~answerCorrect & ~answerConfident) * 3 + ...
                       (~answerCorrect & answerConfident) * 4;

            remindedTrial                 = taskvars.availableForMemProbe(1);
            taskvars.availableForMemProbe = taskvars.availableForMemProbe(2:end);
            trialrec{taskvars.trialIdx}.rwdval = 0.25 * (response < 3) - 0.25 * (response > 2);
        else
            % Fitting. Get the actual response.
            response      = trialrec{taskvars.trialIdx}.choice;
            thisProbe     = find(taskvars.memProbeTrials==taskvars.trialIdx);
            try
                remindedTrial = memRec(thisProbe, 2);
            catch
                % Weird bug where last memprobe wasn't recorded in some early subjects, if it was the last trial.
                % This doesn't matter because there were no choice trials after it.
                remindedTrial = 1;
            end
        end

        %% Update episode cache.

        % Record episode?
        recordEpisode = rand(1)<opts.subjAlphaMem;

        chosenBandit = trialrec{remindedTrial}.choice;

        % If no bandit chosen on reminded trial, don't update cache.
        if (chosenBandit ~= 0 && recordEpisode)
            % If correct & valid, copy the reminded episode to a more recent place in memory.
            % XXX: The horror, the horror. Should this persist? How long should it persist? Separate decay approximates, but...
            if (response == 1 || response == 2)
                taskvars.episodeList{taskvars.trialIdx} = trialrec{remindedTrial};

                taskvars.numBanditEpisodes(chosenBandit) = taskvars.numBanditEpisodes(chosenBandit) + 1;
                taskvars.banditEpisodeList{chosenBandit,taskvars.numBanditEpisodes(chosenBandit)} = trialrec{remindedTrial};
            end

            % If low-confidence, delete the context from the reminded trial.
            if (response == 2)
                taskvars.episodeList{taskvars.trialIdx}.contexts = -1;
                taskvars.banditEpisodeList{chosenBandit,taskvars.numBanditEpisodes(chosenBandit)}.contexts = -1;
            end
        else
            % Don't save anything.
            taskvars.episodeList{taskvars.trialIdx} = [];
            remindedTrial = -1;
        end
    else
        % Invalid probe.
        taskvars.episodeList{taskvars.trialIdx} = [];

        if (opts.simulateSubj)
            response = (answerCorrect & answerConfident) * 4 + ...
                        (answerCorrect & ~answerConfident) * 3 + ...
                        (~answerCorrect & ~answerConfident) * 2 + ...
                         (~answerCorrect & answerConfident) * 1;
            trialrec{taskvars.trialIdx}.rwdval = 0.25 * (response > 2) - 0.25 * (response < 3);
        else
            response = trialrec{taskvars.trialIdx}.choice;
        end

        remindedTrial = -1;
    end

    trialrec{taskvars.trialIdx}.bandits  = [];
    trialrec{taskvars.trialIdx}.choice   = response;
    trialrec{taskvars.trialIdx}.contexts = -1;
    trialrec{taskvars.trialIdx}.probed   = remindedTrial;
end % function doMemProbe
