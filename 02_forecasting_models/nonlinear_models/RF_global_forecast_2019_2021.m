%% RF model, univariate global groundwater forecasting
% Target:
%   dGWS_valid from GLDAS_common_valid_mask_1deg_200401_202412.mat
%
% Framework:
%   Groundwater-history-only, RF model
%
% Input:
%   dGWS-only historical information
%
% Rolling strategy:
%   Round 1: Train 2004-2017, Val 2018, Test 2019
%   Round 2: Train 2004-2018, Val 2019, Test 2020
%   Round 3: Train 2004-2019, Val 2020, Test 2021
%
% Hyperparameter:
%   p_list = [1, 6, 12, 24, 36, 48, 60]
%
% Fixed RF setting:
%   fitrensemble with Method = 'Bag'
%   NumLearningCycles = 50
%   MinLeafSize = 5
%   NumVariablesToSample = max(1, floor(p / 3))
%   numRepeat = 1
%
% Standardization:
%   None. RF is tree-based and uses original lagged dGWS values.
%
% Forecasting strategy:
%   Horizon-specific direct prediction.
%   For each p, train 12 RF models for horizons h = 1, ..., 12.
%
% Final prediction strategy:
%   no-refit-after-validation.
%   First train candidate RF models using the training period.
%   Then select best p using mean validation-year NRMSE across numRepeat.
%   For the test year, keep the RF models estimated from the training period only,
%   and use the observed Train + Val series as the forecasting history.
%   The final test prediction is the monthly mean of numRepeat RF predictions.
%
% Parallel strategy:
%   Use 10 blocks and 15 workers.
%   Save checkpoint after each completed block.
%
% Output:
%   RF_Pred                    [150, 360, 36]
%   RF_NRMSE                   [150, 360, 3]
%   RF_NRMSE_mean              [150, 360]
%   RF_TDA                     [150, 360, 3]
%   RF_TDA_mean                [150, 360]
%   RF_best_p                  [150, 360, 3]
%   RF_Val_NRMSE_best          [150, 360, 3]
%   RF_Val_NRMSE_best_mean     [150, 360]

clc;
clearvars;

%% ===================== User settings =====================

dataFile = 'E:\P1_SKLGB\M_manuscript1\s1_data_results\s1_data_prepare\GLDAS_common_valid_mask_1deg_200401_202412.mat';

outFolder = 'E:\P1_SKLGB\M_manuscript2\s1_data_results\s3_nonlinear_models\';
if ~exist(outFolder, 'dir')
    mkdir(outFolder);
end

% uc1 means uncertainty check
outFile = fullfile(outFolder, 'RF_univariate_global_2019_2021.mat');
checkpointFile = fullfile(outFolder, 'RF_univariate_global_2019_2021_checkpoint.mat');

% If checkpoint file exists, resume unfinished blocks.
% Set false to delete existing checkpoint and restart from block 1.
resumeFromCheckpoint = true;

numWorkers = 15;
block_num = 10;

p_list = [1, 6, 12, 24, 36, 48, 60];
testYears = [2019, 2020, 2021];
horizon = 12;

numRepeat = 1;
% switch randomSeed number for uncertainty check
randomSeed = 20260605;
% randomSeed = 20260719;

rfParam = struct();
rfParam.Method = 'Bag';
rfParam.NumLearningCycles = 50;
rfParam.MinLeafSize = 5;
rfParam.NumVariablesToSampleMode = 'p_over_3';

if exist(checkpointFile, 'file') && ~resumeFromCheckpoint
    delete(checkpointFile);
end

%% ===================== Input file and load data =====================

D = load(dataFile, ...
    'dGWS_valid', ...
    'CommonMask', 'CommonValid', ...
    'validLinearIndex', 'validLat', 'validLon', ...
    'lat', 'lon', 'ym');

dGWS_valid = double(D.dGWS_valid);
CommonMask = D.CommonMask;
CommonValid = logical(D.CommonValid);
validLinearIndex = double(D.validLinearIndex(:));
validLat = double(D.validLat(:));
validLon = double(D.validLon(:));
lat = double(D.lat(:));
lon = double(D.lon(:));
ym = double(D.ym(:));

clear D

[nValid, ~] = size(dGWS_valid);
nLat = numel(lat);
nLon = numel(lon);

fprintf('Valid grid cells: %d\n', nValid);

fprintf('\nRF settings:\n');
fprintf('  Method                     : %s\n', rfParam.Method);
fprintf('  NumLearningCycles          : %d\n', rfParam.NumLearningCycles);
fprintf('  MinLeafSize                : %d\n', rfParam.MinLeafSize);
fprintf('  NumVariablesToSampleMode   : %s\n', rfParam.NumVariablesToSampleMode);
fprintf('  numRepeat                  : %d\n', numRepeat);
fprintf('  randomSeed                 : %d\n', randomSeed);

%% ===================== Initialize or resume valid-grid outputs =====================

nRound = numel(testYears);
nPredMonth = nRound * horizon;

block_size = max(1, ceil(nValid / block_num));
total_blocks = ceil(nValid / block_size);

if resumeFromCheckpoint && exist(checkpointFile, 'file')
    fprintf('\nLoading checkpoint:\n%s\n', checkpointFile);

    C = load(checkpointFile);

    RF_Pred_valid = C.RF_Pred_valid;
    RF_NRMSE_valid = C.RF_NRMSE_valid;
    RF_TDA_valid = C.RF_TDA_valid;
    RF_best_p_valid = C.RF_best_p_valid;
    RF_Val_NRMSE_best_valid = C.RF_Val_NRMSE_best_valid;
    blockDone = C.blockDone;

    if isfield(C, 'cumulativeComputeTimeSeconds')
        cumulativeComputeTimeSeconds = C.cumulativeComputeTimeSeconds;
    else
        cumulativeComputeTimeSeconds = 0;
    end

    clear C
else
    RF_Pred_valid = NaN(nValid, nPredMonth);
    RF_NRMSE_valid = NaN(nValid, nRound);
    RF_TDA_valid = NaN(nValid, nRound);
    RF_best_p_valid = NaN(nValid, nRound);
    RF_Val_NRMSE_best_valid = NaN(nValid, nRound);

    blockDone = false(total_blocks, 1);
    cumulativeComputeTimeSeconds = 0;
end

fprintf('Block number       : %d\n', total_blocks);
fprintf('Approx. block size : %d valid grid cells\n', block_size);

%% ===================== Start parallel pool =====================

poolObj = gcp('nocreate');

if isempty(poolObj)
    poolObj = parpool('local', numWorkers);
elseif poolObj.NumWorkers ~= numWorkers
    delete(poolObj);
    poolObj = parpool('local', numWorkers);
end

Y_valid_const = parallel.pool.Constant(dGWS_valid);

%% ===================== Global RF computation =====================

fprintf('\nStarting RF global computation...\n');

rng(randomSeed, 'twister');

totalTic = tic;
progressTic = tic;

for block = 1:total_blocks
    if blockDone(block)
        fprintf('Block %d/%d already completed. Skipped.\n', block, total_blocks);
        continue;
    end

    start_idx = (block - 1) * block_size + 1;
    end_idx = min(block * block_size, nValid);
    currentValidRows = start_idx:end_idx;
    block_len = numel(currentValidRows);

    blockPred = NaN(block_len, nPredMonth);
    blockNRMSE = NaN(block_len, nRound);
    blockTDA = NaN(block_len, nRound);
    blockBestP = NaN(block_len, nRound);
    blockValNRMSEBest = NaN(block_len, nRound);

    blockTic = tic;

    parfor i = 1:block_len
        globalRow = currentValidRows(i);

        Y_pixel = Y_valid_const.Value(globalRow, :)';

        [pred36, nrmse3, tda3, bestP3, valNRMSEBest3] = runRFOnePixelAllRounds( ...
            Y_pixel, p_list, testYears, horizon, numRepeat, rfParam, randomSeed, globalRow);

        blockPred(i, :) = pred36(:)';
        blockNRMSE(i, :) = nrmse3(:)';
        blockTDA(i, :) = tda3(:)';
        blockBestP(i, :) = bestP3(:)';
        blockValNRMSEBest(i, :) = valNRMSEBest3(:)';
    end

    blockElapsed = toc(blockTic);
    cumulativeComputeTimeSeconds = cumulativeComputeTimeSeconds + blockElapsed;

    RF_Pred_valid(currentValidRows, :) = blockPred;
    RF_NRMSE_valid(currentValidRows, :) = blockNRMSE;
    RF_TDA_valid(currentValidRows, :) = blockTDA;
    RF_best_p_valid(currentValidRows, :) = blockBestP;
    RF_Val_NRMSE_best_valid(currentValidRows, :) = blockValNRMSEBest;

    blockDone(block) = true;

    save(checkpointFile, ...
        'RF_Pred_valid', ...
        'RF_NRMSE_valid', ...
        'RF_TDA_valid', ...
        'RF_best_p_valid', ...
        'RF_Val_NRMSE_best_valid', ...
        'blockDone', ...
        'cumulativeComputeTimeSeconds', ...
        'p_list', 'testYears', 'horizon', ...
        'numRepeat', 'rfParam', 'randomSeed', ...
        'validLinearIndex', ...
        '-v7.3');

    elapsed = toc(progressTic);
    completedBlocks = sum(blockDone);
    progress = completedBlocks / total_blocks * 100;
    estimatedTotal = elapsed / completedBlocks * total_blocks;
    estimatedRemaining = estimatedTotal - elapsed;

    fprintf(['Block %d/%d completed | Block time %.2f min | Progress %.2f%% | ', ...
        'Elapsed %.2f min | Estimated total %.2f min | Remaining %.2f min\n'], ...
        block, total_blocks, blockElapsed / 60, progress, ...
        elapsed / 60, estimatedTotal / 60, estimatedRemaining / 60);
end

elapsedComputeThisRun = toc(totalTic);
fprintf('\nRF global computation finished. Compute time this run: %.2f minutes\n', elapsedComputeThisRun / 60);
fprintf('RF cumulative block computation time: %.2f minutes\n', cumulativeComputeTimeSeconds / 60);

%% ===================== Map valid-grid outputs back to global fields =====================

fprintf('\nMapping valid-grid outputs back to global fields...\n');

RF_Pred = mapValidToGrid3D(RF_Pred_valid, validLinearIndex, nLat, nLon);
RF_NRMSE = mapValidToGrid3D(RF_NRMSE_valid, validLinearIndex, nLat, nLon);
RF_TDA = mapValidToGrid3D(RF_TDA_valid, validLinearIndex, nLat, nLon);
RF_best_p = mapValidToGrid3D(RF_best_p_valid, validLinearIndex, nLat, nLon);
RF_Val_NRMSE_best = mapValidToGrid3D(RF_Val_NRMSE_best_valid, validLinearIndex, nLat, nLon);

RF_NRMSE_mean = mean(RF_NRMSE, 3, 'omitnan');
RF_TDA_mean = mean(RF_TDA, 3, 'omitnan');
RF_Val_NRMSE_best_mean = mean(RF_Val_NRMSE_best, 3, 'omitnan');

predYM = makeYMList(2019, 1, 2021, 12);

yearVec = floor(ym / 100);
monthVec = mod(ym, 100);

%% ===================== Basic summary =====================

fprintf('\nSummary over common valid grid cells:\n');
fprintf('RF_NRMSE_mean: mean = %.4f, median = %.4f, min = %.4f, max = %.4f\n', ...
    mean(RF_NRMSE_mean(CommonValid), 'omitnan'), ...
    median(RF_NRMSE_mean(CommonValid), 'omitnan'), ...
    min(RF_NRMSE_mean(CommonValid), [], 'omitnan'), ...
    max(RF_NRMSE_mean(CommonValid), [], 'omitnan'));

fprintf('RF_TDA_mean: mean = %.2f%%, median = %.2f%%, min = %.2f%%, max = %.2f%%\n', ...
    mean(RF_TDA_mean(CommonValid), 'omitnan'), ...
    median(RF_TDA_mean(CommonValid), 'omitnan'), ...
    min(RF_TDA_mean(CommonValid), [], 'omitnan'), ...
    max(RF_TDA_mean(CommonValid), [], 'omitnan'));

%% ===================== Metadata =====================

Meta = struct();
Meta.model = 'RF';
Meta.framework = 'Groundwater-history-only univariate forecasting';
Meta.input = 'dGWS_valid only';
Meta.period = '2004-01 to 2024-12';
Meta.predictionPeriod = '2019-01 to 2021-12';
Meta.arrayConvention = 'Spatial fields are [lat, lon, time]; RF_Pred time dimension is 2019-01 to 2021-12.';
Meta.rollingStrategy = 'Three rounds: train 2004-2017/2018/2019, validate 2018/2019/2020, test 2019/2020/2021.';
Meta.hyperparameter = 'History window length p selected from p_list by validation-year NRMSE.';
Meta.p_list = p_list;
Meta.rfParam = rfParam;
Meta.numRepeat = numRepeat;
Meta.randomSeed = randomSeed;
Meta.standardization = 'None. RF is tree-based and uses original lagged dGWS values.';
Meta.forecasting = 'Horizon-specific direct prediction with 12 RF models for horizons h = 1,...,12.';
Meta.finalPrediction = 'no-refit-after-validation: RF models are estimated from Train only; Train + Val is used only as observed forecasting history before Test.';
Meta.repeatStrategy = 'Validation p selection uses mean validation NRMSE across numRepeat RF fits. Final test prediction is the monthly mean of numRepeat RF predictions.';
Meta.NRMSE = 'NRMSE = RMSE / range(observed test year).';
Meta.TDA = 'Temporal dynamics agreement, unit: %, calculated using sign agreement of month-to-month changes within the test year.';
Meta.CommonMask = 'Only common valid grid cells from preprocessing are modeled.';
Meta.validGridCount = nValid;
Meta.block_num = block_num;
Meta.numWorkers = numWorkers;
Meta.resumeFromCheckpoint = resumeFromCheckpoint;
Meta.computeTimeMinutes = cumulativeComputeTimeSeconds / 60;
Meta.computeTimeHours = cumulativeComputeTimeSeconds / 3600;
Meta.computeTimeThisRunMinutes = elapsedComputeThisRun / 60;
Meta.computeTimeThisRunHours = elapsedComputeThisRun / 3600;
Meta.computeTimeNote = 'computeTimeMinutes and computeTimeHours are cumulative block-wise parallel computation time accumulated from checkpoint; mapping, summary, plotting, and final saving are excluded.';
Meta.note1 = 'Validation-year NRMSE is used only for p selection.';
Meta.note2 = 'RF_Val_NRMSE_best is retained for traceability of validation-based p selection.';
Meta.note3 = 'numRepeat is retained as a user-adjustable setting; current default is 1 for computational feasibility.';

%% ===================== Save final output =====================

fprintf('\nSaving final output to:\n%s\n', outFile);

save(outFile, ...
    'RF_Pred', ...
    'RF_NRMSE', 'RF_NRMSE_mean', ...
    'RF_TDA', 'RF_TDA_mean', ...
    'RF_best_p', ...
    'RF_Val_NRMSE_best', 'RF_Val_NRMSE_best_mean', ...
    'p_list', 'testYears', 'horizon', ...
    'numRepeat', 'rfParam', 'randomSeed', 'predYM', ...
    'CommonMask', 'CommonValid', 'validLinearIndex', 'nValid', ...
    'validLat', 'validLon', ...
    'lat', 'lon', 'ym', 'yearVec', 'monthVec', ...
    'Meta', ...
    '-v7.3');

fprintf('Final output saved.\n');

%% ===================== Diagnostic plot =====================

figure;

p = pcolor(lon, lat, RF_NRMSE_mean);
set(p, 'EdgeColor', 'none');
shading flat;

set(gca, 'YDir', 'normal');
axis equal tight;
box on;
grid on;
clim([0,0.4]);

colormap(jet);
cb = colorbar;
ylabel(cb, 'NRMSE');

title('RF mean NRMSE, testing years 2019-2021');
xlabel('Longitude');
ylabel('Latitude');

set(gca, 'FontName', 'Times New Roman', 'FontSize', 11);

try
    set(p, 'AlphaData', ~isnan(RF_NRMSE_mean));
catch
end

%% ===================== Local functions =====================

function [pred36, nrmse3, tda3, bestP3, valNRMSEBest3] = runRFOnePixelAllRounds( ...
    Y, p_list, testYears, horizon, numRepeat, rfParam, randomSeed, pointID)

    Y = double(Y(:));

    nRound = numel(testYears);

    pred36 = NaN(nRound * horizon, 1);
    nrmse3 = NaN(nRound, 1);
    tda3 = NaN(nRound, 1);
    bestP3 = NaN(nRound, 1);
    valNRMSEBest3 = NaN(nRound, 1);

    if numel(Y) ~= 252
        return;
    end

    for r = 1:nRound
        testYear = testYears(r);

        % Example:
        % testYear = 2019:
        % trainEndIdx = 168 -> 2004-01 to 2017-12
        % valIdx      = 169:180 -> 2018
        % testIdx     = 181:192 -> 2019
        trainEndIdx = (testYear - 2005) * 12;

        valIdx = (trainEndIdx + 1):(trainEndIdx + 12);
        testIdx = (trainEndIdx + 13):(trainEndIdx + 24);

        Y_train = Y(1:trainEndIdx);
        Y_val = Y(valIdx);
        Y_test = Y(testIdx);

        valNRMSE_mean_by_p = Inf(numel(p_list), 1);

        for k = 1:numel(p_list)
            p = p_list(k);

            if numel(Y_train) <= p + horizon - 1
                continue;
            end

            valNRMSE_repeat = NaN(numRepeat, 1);

            for rep = 1:numRepeat
                try
                    seedVal = makeSeed(randomSeed, pointID, r, k, rep, 0);

                    rfPack = fitRFHorizonSpecificFromSeries( ...
                        Y_train, p, horizon, rfParam, seedVal);

                    initWindowVal = Y_train(end-p+1:end);
                    Y_val_pred = predictRFHorizonSpecific(rfPack, initWindowVal);

                    valNRMSE_repeat(rep) = calcNRMSE(Y_val, Y_val_pred);
                catch
                    valNRMSE_repeat(rep) = NaN;
                end
            end

            tmpMean = mean(valNRMSE_repeat, 'omitnan');

            if isfinite(tmpMean)
                valNRMSE_mean_by_p(k) = tmpMean;
            else
                valNRMSE_mean_by_p(k) = Inf;
            end
        end

        [valBest, bestIdx] = min(valNRMSE_mean_by_p);

        if isempty(bestIdx) || ~isfinite(valBest)
            continue;
        end

        bestP = p_list(bestIdx);

        % No-refit-after-validation:
        % keep the RF models estimated from Train only.
        % Use Train + Val only as the observed forecasting history before Test.
        Y_train_val = Y(1:(trainEndIdx + 12));

        Y_test_pred_repeat = NaN(horizon, numRepeat);

        for rep = 1:numRepeat
            try
                seedTest = makeSeed(randomSeed, pointID, r, bestIdx, rep, 1);

                finalPack = fitRFHorizonSpecificFromSeries( ...
                    Y_train, bestP, horizon, rfParam, seedTest);

                initWindowTest = Y_train_val(end-bestP+1:end);
                Y_test_pred_repeat(:, rep) = predictRFHorizonSpecific(finalPack, initWindowTest);
            catch
                Y_test_pred_repeat(:, rep) = NaN;
            end
        end

        Y_test_pred = mean(Y_test_pred_repeat, 2, 'omitnan');

        if numel(Y_test_pred) ~= horizon || any(~isfinite(Y_test_pred))
            continue;
        end

        predRange = ((r - 1) * horizon + 1):(r * horizon);

        pred36(predRange) = Y_test_pred(:);
        nrmse3(r) = calcNRMSE(Y_test, Y_test_pred);
        tda3(r) = calcTDA(Y_test, Y_test_pred);
        bestP3(r) = bestP;
        valNRMSEBest3(r) = valBest;
    end
end

function rfPack = fitRFHorizonSpecificFromSeries(Y_series, p, horizon, rfParam, baseSeed)
    Y_series = double(Y_series(:));

    modelCell = cell(horizon, 1);

    for h = 1:horizon
        [X, Y_target] = buildLaggedSamplesDirect(Y_series, p, h);

        numVariablesToSample = getNumVariablesToSample(p, rfParam);

        seedH = mod(baseSeed + h, 2^32 - 1);
        if seedH == 0
            seedH = 1;
        end

        rng(seedH, 'twister');

        treeTemplate = templateTree( ...
            'MinLeafSize', rfParam.MinLeafSize, ...
            'NumVariablesToSample', numVariablesToSample);

        Mdl = fitrensemble( ...
            X, Y_target, ...
            'Method', rfParam.Method, ...
            'NumLearningCycles', rfParam.NumLearningCycles, ...
            'Learners', treeTemplate);

        modelCell{h} = Mdl;
    end

    rfPack = struct();
    rfPack.modelCell = modelCell;
    rfPack.p = p;
    rfPack.horizon = horizon;
    rfPack.rfParam = rfParam;
end

function [X, Y_target] = buildLaggedSamplesDirect(Y_series, p, h)
    Y_series = double(Y_series(:));

    nSample = numel(Y_series) - p - h + 1;

    if nSample <= 0
        error('No direct samples can be constructed.');
    end

    X = zeros(nSample, p);
    Y_target = zeros(nSample, 1);

    for j = 1:nSample
        X(j, :) = Y_series(j:j+p-1)';
        Y_target(j) = Y_series(j+p+h-1);
    end
end

function Y_pred = predictRFHorizonSpecific(rfPack, initialWindow)
    p = rfPack.p;
    horizon = rfPack.horizon;

    initialWindow = double(initialWindow(:)');

    if numel(initialWindow) ~= p
        error('Initial window length is %d, expected p = %d.', numel(initialWindow), p);
    end

    Y_pred = NaN(horizon, 1);

    for h = 1:horizon
        Mdl = rfPack.modelCell{h};

        X_new = initialWindow;
        y_pred = predict(Mdl, X_new);
        y_pred = double(y_pred(1));

        Y_pred(h) = y_pred;
    end
end

function numVariablesToSample = getNumVariablesToSample(p, rfParam)
    switch rfParam.NumVariablesToSampleMode
        case 'p_over_3'
            numVariablesToSample = max(1, floor(p / 3));

        case 'sqrt'
            numVariablesToSample = max(1, floor(sqrt(p)));

        case 'all'
            numVariablesToSample = p;

        otherwise
            error('Unknown NumVariablesToSampleMode: %s', ...
                rfParam.NumVariablesToSampleMode);
    end

    numVariablesToSample = min(numVariablesToSample, p);
end

function seedVal = makeSeed(randomSeed, pointID, roundID, pID, repeatID, stageID)
    seedRaw = randomSeed ...
        + pointID * 100 ...
        + roundID * 10000 ...
        + pID * 1000 ...
        + repeatID * 10 ...
        + stageID;

    seedVal = mod(seedRaw, 2^32 - 1);

    if seedVal == 0
        seedVal = randomSeed;
    end
end

function nrmse = calcNRMSE(obs, pred)
    obs = double(obs(:));
    pred = double(pred(:));

    if numel(obs) ~= numel(pred) || isempty(obs)
        nrmse = NaN;
        return;
    end

    residual = pred - obs;
    rmse = sqrt(mean(residual .^ 2));
    denom = max(obs) - min(obs);

    if denom == 0
        if rmse == 0
            nrmse = 0;
        else
            nrmse = Inf;
        end
    else
        nrmse = rmse / denom;
    end
end

function tda = calcTDA(obs, pred)
    obs = double(obs(:));
    pred = double(pred(:));

    if numel(obs) ~= numel(pred) || numel(obs) < 2
        tda = NaN;
        return;
    end

    actualDiff = diff(obs);
    predDiff = diff(pred);

    tda = mean(sign(actualDiff) == sign(predDiff)) * 100;
end

function Ygrid = mapValidToGrid3D(Yvalid, validLinearIndex, nLat, nLon)
    [~, nLayer] = size(Yvalid);

    Ygrid = NaN(nLat, nLon, nLayer);

    for k = 1:nLayer
        tmp = NaN(nLat, nLon);
        tmp(validLinearIndex) = Yvalid(:, k);
        Ygrid(:, :, k) = tmp;
    end
end

function ym = makeYMList(y1, m1, y2, m2)
    d1 = datetime(y1, m1, 1);
    d2 = datetime(y2, m2, 1);
    d = d1:calmonths(1):d2;
    ym = year(d(:)) * 100 + month(d(:));
    ym = double(ym(:));
end