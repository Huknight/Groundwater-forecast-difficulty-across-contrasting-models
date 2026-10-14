%% RFX model, multivariate global groundwater forecasting
% Target:
%   dGWS_valid from GLDAS_common_valid_mask_1deg_200401_202412.mat
%
% Framework:
%   Multivariate historical-input RF model
%
% Input:
%   Historical dGWS + historical exogenous variables:
%   dGWS_valid, Prec_valid, Evap_valid, Runoff_valid, dSM2m_valid, SurfT_valid
%
% Rolling strategy:
%   Round 1: Train 2004-2017, Val 2018, Test 2019
%   Round 2: Train 2004-2018, Val 2019, Test 2020
%   Round 3: Train 2004-2019, Val 2020, Test 2021
%
% Hyperparameters:
%   p_list = [1, 6, 12, 24, 36, 48, 60]
%       p controls dGWS historical window length.
%
%   x_list = [1, 3, 6]
%       x controls exogenous-variable historical window length.
%
% RF setting:
%   fitrensemble with Method = 'Bag'
%   NumLearningCycles = 50
%   MinLeafSize = 5
%   NumVariablesToSample = max(1, floor((p + numExogVars * x) / 3))
%   numRepeat = 1
%
% Standardization:
%   None. RF is tree-based and uses original lagged values.
%
% Forecasting strategy:
%   Horizon-specific direct prediction.
%   For each (p, x), train 12 RF models for horizons h = 1, ..., 12.
%
% No future exogenous variables:
%   Prediction input window ends at the month before the forecast year.
%
% Final prediction strategy:
%   no-refit-after-validation.
%   First train candidate RFX models using the training period.
%   Then select best (p, x) using mean validation-year NRMSE across numRepeat.
%   For the test year, keep the RFX models estimated from the training period only,
%   and use observed Train + Val multivariate history as the forecasting input.
%   The final test prediction is the monthly mean of numRepeat RFX predictions.
%
% Parallel strategy:
%   Use 10 blocks and 15 workers.
%   Save checkpoint after each completed block.
%
% Output:
%   RFX_Pred                    [150, 360, 36]
%   RFX_NRMSE                   [150, 360, 3]
%   RFX_NRMSE_mean              [150, 360]
%   RFX_TDA                     [150, 360, 3]
%   RFX_TDA_mean                [150, 360]
%   RFX_best_p                  [150, 360, 3]
%   RFX_best_x                  [150, 360, 3]
%   RFX_Val_NRMSE_best          [150, 360, 3]
%   RFX_Val_NRMSE_best_mean     [150, 360]

clc;
clearvars;

%% ===================== User settings =====================

dataFile = 'E:\P1_SKLGB\M_manuscript1\s1_data_results\s1_data_prepare\GLDAS_common_valid_mask_1deg_200401_202412.mat';

outFolder = 'E:\P1_SKLGB\M_manuscript2\s1_data_results\s4_nonlinear_x_models\';
if ~exist(outFolder, 'dir')
    mkdir(outFolder);
end

% uc1 means uncertainty check
outFile = fullfile(outFolder, 'RFX_multivariate_global_2019_2021.mat');
checkpointFile = fullfile(outFolder, 'RFX_multivariate_global_2019_2021_checkpoint.mat');

% If checkpoint file exists, resume unfinished blocks.
% Set false to delete existing checkpoint and restart from block 1.
resumeFromCheckpoint = true;

numWorkers = 15;
block_num = 10;

p_list = [1, 6, 12, 24, 36, 48, 60];
x_list = [1, 3, 6];

testYears = [2019, 2020, 2021];
horizon = 12;

numRepeat = 1;
% switch randomSeed number for uncertainty check
randomSeed = 20260605;
% randomSeed = 20260719;

inputVarNames = {'dGWS', 'Prec', 'Evap', 'Runoff', 'dSM2m', 'SurfT'};
exogVarNames = {'Prec', 'Evap', 'Runoff', 'dSM2m', 'SurfT'};

rfParam = struct();
rfParam.Method = 'Bag';
rfParam.NumLearningCycles = 50;
rfParam.MinLeafSize = 5;
rfParam.NumVariablesToSampleMode = 'p_x_over_3';

if exist(checkpointFile, 'file') && ~resumeFromCheckpoint
    delete(checkpointFile);
end

%% ===================== Input file and load data =====================

D = load(dataFile, ...
    'dGWS_valid', ...
    'Prec_valid', 'Evap_valid', 'Runoff_valid', 'dSM2m_valid', 'SurfT_valid', ...
    'CommonMask', 'CommonValid', ...
    'validLinearIndex', 'validLat', 'validLon', ...
    'lat', 'lon', 'ym');

dGWS_valid = double(D.dGWS_valid);
Prec_valid = double(D.Prec_valid);
Evap_valid = double(D.Evap_valid);
Runoff_valid = double(D.Runoff_valid);
dSM2m_valid = double(D.dSM2m_valid);
SurfT_valid = double(D.SurfT_valid);

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

fprintf('\nRFX settings:\n');
fprintf('  Input variables             : %s\n', strjoin(inputVarNames, ', '));
fprintf('  Exogenous variables         : %s\n', strjoin(exogVarNames, ', '));
fprintf('  p_list                      : %s\n', mat2str(p_list));
fprintf('  x_list                      : %s\n', mat2str(x_list));
fprintf('  Method                      : %s\n', rfParam.Method);
fprintf('  NumLearningCycles           : %d\n', rfParam.NumLearningCycles);
fprintf('  MinLeafSize                 : %d\n', rfParam.MinLeafSize);
fprintf('  NumVariablesToSampleMode    : %s\n', rfParam.NumVariablesToSampleMode);
fprintf('  numRepeat                   : %d\n', numRepeat);
fprintf('  randomSeed                  : %d\n', randomSeed);

%% ===================== Initialize or resume valid-grid outputs =====================

nRound = numel(testYears);
nPredMonth = nRound * horizon;

block_size = max(1, ceil(nValid / block_num));
total_blocks = ceil(nValid / block_size);

if resumeFromCheckpoint && exist(checkpointFile, 'file')
    fprintf('\nLoading checkpoint:\n%s\n', checkpointFile);

    C = load(checkpointFile);

    RFX_Pred_valid = C.RFX_Pred_valid;
    RFX_NRMSE_valid = C.RFX_NRMSE_valid;
    RFX_TDA_valid = C.RFX_TDA_valid;
    RFX_best_p_valid = C.RFX_best_p_valid;
    RFX_best_x_valid = C.RFX_best_x_valid;
    RFX_Val_NRMSE_best_valid = C.RFX_Val_NRMSE_best_valid;
    blockDone = C.blockDone;

    if isfield(C, 'cumulativeComputeTimeSeconds')
        cumulativeComputeTimeSeconds = C.cumulativeComputeTimeSeconds;
    else
        cumulativeComputeTimeSeconds = 0;
    end

    clear C
else
    RFX_Pred_valid = NaN(nValid, nPredMonth);
    RFX_NRMSE_valid = NaN(nValid, nRound);
    RFX_TDA_valid = NaN(nValid, nRound);
    RFX_best_p_valid = NaN(nValid, nRound);
    RFX_best_x_valid = NaN(nValid, nRound);
    RFX_Val_NRMSE_best_valid = NaN(nValid, nRound);

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

dGWS_const = parallel.pool.Constant(dGWS_valid);
Prec_const = parallel.pool.Constant(Prec_valid);
Evap_const = parallel.pool.Constant(Evap_valid);
Runoff_const = parallel.pool.Constant(Runoff_valid);
dSM2m_const = parallel.pool.Constant(dSM2m_valid);
SurfT_const = parallel.pool.Constant(SurfT_valid);

%% ===================== Global RFX computation =====================

fprintf('\nStarting RFX global computation...\n');

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
    blockBestX = NaN(block_len, nRound);
    blockValNRMSEBest = NaN(block_len, nRound);

    blockTic = tic;

    parfor i = 1:block_len
        globalRow = currentValidRows(i);

        X_all = [
            dGWS_const.Value(globalRow, :)', ...
            Prec_const.Value(globalRow, :)', ...
            Evap_const.Value(globalRow, :)', ...
            Runoff_const.Value(globalRow, :)', ...
            dSM2m_const.Value(globalRow, :)', ...
            SurfT_const.Value(globalRow, :)'
        ];

        Y_pixel = dGWS_const.Value(globalRow, :)';

        [pred36, nrmse3, tda3, bestP3, bestX3, valNRMSEBest3] = runRFXOnePixelAllRounds( ...
            X_all, Y_pixel, p_list, x_list, testYears, horizon, numRepeat, rfParam, randomSeed, globalRow);

        blockPred(i, :) = pred36(:)';
        blockNRMSE(i, :) = nrmse3(:)';
        blockTDA(i, :) = tda3(:)';
        blockBestP(i, :) = bestP3(:)';
        blockBestX(i, :) = bestX3(:)';
        blockValNRMSEBest(i, :) = valNRMSEBest3(:)';
    end

    blockElapsed = toc(blockTic);
    cumulativeComputeTimeSeconds = cumulativeComputeTimeSeconds + blockElapsed;

    RFX_Pred_valid(currentValidRows, :) = blockPred;
    RFX_NRMSE_valid(currentValidRows, :) = blockNRMSE;
    RFX_TDA_valid(currentValidRows, :) = blockTDA;
    RFX_best_p_valid(currentValidRows, :) = blockBestP;
    RFX_best_x_valid(currentValidRows, :) = blockBestX;
    RFX_Val_NRMSE_best_valid(currentValidRows, :) = blockValNRMSEBest;

    blockDone(block) = true;

    save(checkpointFile, ...
        'RFX_Pred_valid', ...
        'RFX_NRMSE_valid', ...
        'RFX_TDA_valid', ...
        'RFX_best_p_valid', ...
        'RFX_best_x_valid', ...
        'RFX_Val_NRMSE_best_valid', ...
        'blockDone', ...
        'cumulativeComputeTimeSeconds', ...
        'p_list', 'x_list', 'testYears', 'horizon', ...
        'numRepeat', 'rfParam', 'inputVarNames', 'exogVarNames', 'randomSeed', ...
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
fprintf('\nRFX global computation finished. Compute time this run: %.2f minutes\n', elapsedComputeThisRun / 60);
fprintf('RFX cumulative block computation time: %.2f minutes\n', cumulativeComputeTimeSeconds / 60);

%% ===================== Map valid-grid outputs back to global fields =====================

fprintf('\nMapping valid-grid outputs back to global fields...\n');

RFX_Pred = mapValidToGrid3D(RFX_Pred_valid, validLinearIndex, nLat, nLon);
RFX_NRMSE = mapValidToGrid3D(RFX_NRMSE_valid, validLinearIndex, nLat, nLon);
RFX_TDA = mapValidToGrid3D(RFX_TDA_valid, validLinearIndex, nLat, nLon);
RFX_best_p = mapValidToGrid3D(RFX_best_p_valid, validLinearIndex, nLat, nLon);
RFX_best_x = mapValidToGrid3D(RFX_best_x_valid, validLinearIndex, nLat, nLon);
RFX_Val_NRMSE_best = mapValidToGrid3D(RFX_Val_NRMSE_best_valid, validLinearIndex, nLat, nLon);

RFX_NRMSE_mean = mean(RFX_NRMSE, 3, 'omitnan');
RFX_TDA_mean = mean(RFX_TDA, 3, 'omitnan');
RFX_Val_NRMSE_best_mean = mean(RFX_Val_NRMSE_best, 3, 'omitnan');

predYM = makeYMList(2019, 1, 2021, 12);

yearVec = floor(ym / 100);
monthVec = mod(ym, 100);

%% ===================== Basic summary =====================

fprintf('\nSummary over common valid grid cells:\n');
fprintf('RFX_NRMSE_mean: mean = %.4f, median = %.4f, min = %.4f, max = %.4f\n', ...
    mean(RFX_NRMSE_mean(CommonValid), 'omitnan'), ...
    median(RFX_NRMSE_mean(CommonValid), 'omitnan'), ...
    min(RFX_NRMSE_mean(CommonValid), [], 'omitnan'), ...
    max(RFX_NRMSE_mean(CommonValid), [], 'omitnan'));

fprintf('RFX_TDA_mean: mean = %.2f%%, median = %.2f%%, min = %.2f%%, max = %.2f%%\n', ...
    mean(RFX_TDA_mean(CommonValid), 'omitnan'), ...
    median(RFX_TDA_mean(CommonValid), 'omitnan'), ...
    min(RFX_TDA_mean(CommonValid), [], 'omitnan'), ...
    max(RFX_TDA_mean(CommonValid), [], 'omitnan'));

%% ===================== Metadata =====================

Meta = struct();
Meta.model = 'RFX';
Meta.framework = 'Multivariate historical-input groundwater forecasting';
Meta.input = 'Historical dGWS_valid, Prec_valid, Evap_valid, Runoff_valid, dSM2m_valid, and SurfT_valid.';
Meta.target = 'Future dGWS_valid.';
Meta.inputVarNames = inputVarNames;
Meta.exogVarNames = exogVarNames;
Meta.period = '2004-01 to 2024-12';
Meta.predictionPeriod = '2019-01 to 2021-12';
Meta.arrayConvention = 'Spatial fields are [lat, lon, time]; RFX_Pred time dimension is 2019-01 to 2021-12.';
Meta.rollingStrategy = 'Three rounds: train 2004-2017/2018/2019, validate 2018/2019/2020, test 2019/2020/2021.';
Meta.hyperparameter = 'Two-window search: p controls dGWS history length; x controls exogenous-variable history length. Best (p,x) is selected by validation-year NRMSE.';
Meta.p_list = p_list;
Meta.x_list = x_list;
Meta.rfParam = rfParam;
Meta.numRepeat = numRepeat;
Meta.randomSeed = randomSeed;
Meta.standardization = 'None. RF is tree-based and uses original lagged values.';
Meta.forecasting = 'Horizon-specific direct prediction with 12 RF models for horizons h = 1,...,12 for each candidate (p,x).';
Meta.noFutureExogenousVariables = 'Prediction input uses previous-p-month dGWS history and previous-x-month exogenous-variable history ending before the forecast year; future exogenous variables are not used.';
Meta.finalPrediction = 'no-refit-after-validation: RFX models are estimated from Train only; Train + Val is used only as observed multivariate forecasting history before Test.';
Meta.NumVariablesToSample = 'Computed from total predictor count nPredictor = p + numExogVars*x; default max(1,floor(nPredictor/3)).';
Meta.repeatStrategy = 'Validation (p,x) selection uses mean validation NRMSE across numRepeat RFX fits. Final test prediction is the monthly mean of numRepeat RFX predictions.';
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
Meta.note1 = 'RFX_Val_NRMSE_best is retained for traceability of validation-based (p,x) selection.';
Meta.note2 = 'numRepeat is retained as a user-adjustable setting; current default is 1 for computational feasibility.';

%% ===================== Save final output =====================

fprintf('\nSaving final output to:\n%s\n', outFile);

save(outFile, ...
    'RFX_Pred', ...
    'RFX_NRMSE', 'RFX_NRMSE_mean', ...
    'RFX_TDA', 'RFX_TDA_mean', ...
    'RFX_best_p', 'RFX_best_x', ...
    'RFX_Val_NRMSE_best', 'RFX_Val_NRMSE_best_mean', ...
    'p_list', 'x_list', 'testYears', 'horizon', ...
    'numRepeat', 'rfParam', 'inputVarNames', 'exogVarNames', 'randomSeed', 'predYM', ...
    'CommonMask', 'CommonValid', 'validLinearIndex', 'nValid', ...
    'validLat', 'validLon', ...
    'lat', 'lon', 'ym', 'yearVec', 'monthVec', ...
    'Meta', ...
    '-v7.3');

fprintf('Final output saved.\n');

%% ===================== Diagnostic plot =====================

figure;

p = pcolor(lon, lat, RFX_NRMSE_mean);
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

title('RFX mean NRMSE, testing years 2019-2021');
xlabel('Longitude');
ylabel('Latitude');

set(gca, 'FontName', 'Times New Roman', 'FontSize', 11);

try
    set(p, 'AlphaData', ~isnan(RFX_NRMSE_mean));
catch
end

%% ===================== Local functions =====================

function [pred36, nrmse3, tda3, bestP3, bestX3, valNRMSEBest3] = runRFXOnePixelAllRounds( ...
    X_all, Y, p_list, x_list, testYears, horizon, numRepeat, rfParam, randomSeed, pointID)

    X_all = double(X_all);
    Y = double(Y(:));

    nRound = numel(testYears);

    pred36 = NaN(nRound * horizon, 1);
    nrmse3 = NaN(nRound, 1);
    tda3 = NaN(nRound, 1);
    bestP3 = NaN(nRound, 1);
    bestX3 = NaN(nRound, 1);
    valNRMSEBest3 = NaN(nRound, 1);

    if size(X_all, 1) ~= 252 || numel(Y) ~= 252
        return;
    end

    if size(X_all, 2) < 2
        return;
    end

    for r = 1:nRound
        testYear = testYears(r);

        trainEndIdx = (testYear - 2005) * 12;

        valIdx = (trainEndIdx + 1):(trainEndIdx + horizon);
        testIdx = (trainEndIdx + horizon + 1):(trainEndIdx + 2 * horizon);

        X_train_all = X_all(1:trainEndIdx, :);
        X_train_val_all = X_all(1:(trainEndIdx + horizon), :);

        Y_train = Y(1:trainEndIdx);
        Y_val = Y(valIdx);
        Y_test = Y(testIdx);

        valNRMSE_mean_by_combo = Inf(numel(p_list), numel(x_list));

        for ip = 1:numel(p_list)
            p = p_list(ip);

            for ix = 1:numel(x_list)
                x = x_list(ix);

                maxLag = max(p, x);

                if size(X_train_all, 1) <= maxLag + horizon - 1
                    continue;
                end

                valNRMSE_repeat = NaN(numRepeat, 1);

                for rep = 1:numRepeat
                    try
                        seedVal = makeSeed(randomSeed, pointID, r, ip, ix, rep, 0);

                        rfPack = fitRFXHorizonSpecificFromSeries( ...
                            X_train_all, Y_train, p, x, horizon, rfParam, seedVal);

                        initWindowVal = X_train_all(end-maxLag+1:end, :);
                        Y_val_pred = predictRFXHorizonSpecific(rfPack, initWindowVal);

                        valNRMSE_repeat(rep) = calcNRMSE(Y_val, Y_val_pred);
                    catch
                        valNRMSE_repeat(rep) = NaN;
                    end
                end

                tmpMean = mean(valNRMSE_repeat, 'omitnan');

                if isfinite(tmpMean)
                    valNRMSE_mean_by_combo(ip, ix) = tmpMean;
                else
                    valNRMSE_mean_by_combo(ip, ix) = Inf;
                end
            end
        end

        [valBest, linearBestIdx] = min(valNRMSE_mean_by_combo(:));

        if isempty(linearBestIdx) || ~isfinite(valBest)
            continue;
        end

        [bestIp, bestIx] = ind2sub(size(valNRMSE_mean_by_combo), linearBestIdx);

        bestP = p_list(bestIp);
        bestX = x_list(bestIx);
        bestMaxLag = max(bestP, bestX);

        Y_test_pred_repeat = NaN(horizon, numRepeat);

        for rep = 1:numRepeat
            try
                seedTest = makeSeed(randomSeed, pointID, r, bestIp, bestIx, rep, 1);

                finalPack = fitRFXHorizonSpecificFromSeries( ...
                    X_train_all, Y_train, bestP, bestX, horizon, rfParam, seedTest);

                initWindowTest = X_train_val_all(end-bestMaxLag+1:end, :);
                Y_test_pred_repeat(:, rep) = predictRFXHorizonSpecific(finalPack, initWindowTest);
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
        bestX3(r) = bestX;
        valNRMSEBest3(r) = valBest;
    end
end

function rfPack = fitRFXHorizonSpecificFromSeries( ...
    X_all, Y_series, p, x, horizon, rfParam, baseSeed)

    X_all = double(X_all);
    Y_series = double(Y_series(:));

    modelCell = cell(horizon, 1);

    nInputVars = size(X_all, 2);
    nExogVars = nInputVars - 1;
    nPredictor = p + nExogVars * x;

    for h = 1:horizon
        [X_train, Y_target] = buildRFXLaggedSamplesDirect(X_all, Y_series, p, x, h);

        numVariablesToSample = getNumVariablesToSample(nPredictor, rfParam);

        seedH = mod(baseSeed + h, 2^32 - 1);
        if seedH == 0
            seedH = 1;
        end

        rng(seedH, 'twister');

        treeTemplate = templateTree( ...
            'MinLeafSize', rfParam.MinLeafSize, ...
            'NumVariablesToSample', numVariablesToSample);

        Mdl = fitrensemble( ...
            X_train, Y_target, ...
            'Method', rfParam.Method, ...
            'NumLearningCycles', rfParam.NumLearningCycles, ...
            'Learners', treeTemplate);

        modelCell{h} = Mdl;
    end

    rfPack = struct();
    rfPack.modelCell = modelCell;
    rfPack.p = p;
    rfPack.x = x;
    rfPack.horizon = horizon;
    rfPack.nInputVars = nInputVars;
    rfPack.nExogVars = nExogVars;
    rfPack.nPredictor = nPredictor;
    rfPack.rfParam = rfParam;
end

function [X_train, Y_target] = buildRFXLaggedSamplesDirect(X_all, Y_series, p, x, h)
    X_all = double(X_all);
    Y_series = double(Y_series(:));

    nTime = size(X_all, 1);
    nInputVars = size(X_all, 2);

    nExogVars = nInputVars - 1;
    nPredictor = p + nExogVars * x;

    maxLag = max(p, x);

    nSample = nTime - maxLag - h + 1;

    if nSample <= 0
        error('No RFX direct samples can be constructed.');
    end

    X_train = zeros(nSample, nPredictor);
    Y_target = zeros(nSample, 1);

    for j = 1:nSample
        anchorIdx = j + maxLag - 1;

        dGWS_lag = X_all((anchorIdx - p + 1):anchorIdx, 1)';

        exogFeat = zeros(1, nExogVars * x);
        pos = 1;

        for v = 2:nInputVars
            exog_lag = X_all((anchorIdx - x + 1):anchorIdx, v)';
            exogFeat(pos:(pos + x - 1)) = exog_lag;
            pos = pos + x;
        end

        X_train(j, :) = [dGWS_lag, exogFeat];
        Y_target(j) = Y_series(anchorIdx + h);
    end
end

function Y_pred = predictRFXHorizonSpecific(rfPack, initialWindow)
    p = rfPack.p;
    x = rfPack.x;
    horizon = rfPack.horizon;

    initialWindow = double(initialWindow);

    maxLag = max(p, x);

    if size(initialWindow, 1) ~= maxLag
        error('Initial window length is %d, expected max(p,x) = %d.', ...
            size(initialWindow, 1), maxLag);
    end

    if size(initialWindow, 2) ~= rfPack.nInputVars
        error('Initial window has %d variables, expected %d.', ...
            size(initialWindow, 2), rfPack.nInputVars);
    end

    X_new = makeRFXOneFeatureRow(initialWindow, p, x);

    Y_pred = NaN(horizon, 1);

    for h = 1:horizon
        Mdl = rfPack.modelCell{h};

        y_pred = predict(Mdl, X_new);
        y_pred = double(y_pred(1));

        Y_pred(h) = y_pred;
    end
end

function X_new = makeRFXOneFeatureRow(initialWindow, p, x)
    initialWindow = double(initialWindow);

    nInputVars = size(initialWindow, 2);
    nExogVars = nInputVars - 1;

    nPredictor = p + nExogVars * x;
    X_new = zeros(1, nPredictor);

    maxLag = max(p, x);
    anchorIdx = maxLag;

    dGWS_lag = initialWindow((anchorIdx - p + 1):anchorIdx, 1)';
    X_new(1:p) = dGWS_lag;

    pos = p + 1;

    for v = 2:nInputVars
        exog_lag = initialWindow((anchorIdx - x + 1):anchorIdx, v)';
        X_new(pos:(pos + x - 1)) = exog_lag;
        pos = pos + x;
    end
end

function numVariablesToSample = getNumVariablesToSample(nPredictor, rfParam)
    switch rfParam.NumVariablesToSampleMode
        case 'p_x_over_3'
            numVariablesToSample = max(1, floor(nPredictor / 3));

        case 'sqrt'
            numVariablesToSample = max(1, floor(sqrt(nPredictor)));

        case 'all'
            numVariablesToSample = nPredictor;

        otherwise
            error('Unknown NumVariablesToSampleMode: %s', ...
                rfParam.NumVariablesToSampleMode);
    end

    numVariablesToSample = min(numVariablesToSample, nPredictor);
end

function seedVal = makeSeed(randomSeed, pointID, roundID, pID, xID, repeatID, stageID)
    seedRaw = randomSeed ...
        + pointID * 100 ...
        + roundID * 10000 ...
        + pID * 1000 ...
        + xID * 100 ...
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