%% AR model, univariate global groundwater forecasting
% Target:
%   dGWS_valid from GLDAS_common_valid_mask_1deg_200401_202412.mat
%
% Framework:
%   Groundwater-history-only, AR model
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
% Final prediction strategy:
%   no-refit-after-validation.
%   First train candidate AR models using the training period.
%   Then select best p using validation-year NRMSE.
%   For the test year, keep the AR model estimated from the training period only,
%   and use the observed Train + Val series as the forecasting history.
%
% Parallel strategy:
%   Use 10 blocks and 15 workers.
%   Save checkpoint after each completed block.
%
% Output:
%   AR_Pred                    [150, 360, 36]
%   AR_NRMSE                   [150, 360, 3]
%   AR_NRMSE_mean              [150, 360]
%   AR_TDA                     [150, 360, 3]
%   AR_TDA_mean                [150, 360]
%   AR_best_p                  [150, 360, 3]
%   AR_Val_NRMSE_best          [150, 360, 3]
%   AR_Val_NRMSE_best_mean     [150, 360]

clc;
clearvars;

%% ===================== User settings =====================

dataFile = 'E:\P1_SKLGB\M_manuscript1\s1_data_results\s1_data_prepare\GLDAS_common_valid_mask_1deg_200401_202412.mat';

outFolder = 'E:\P1_SKLGB\M_manuscript2\s1_data_results\s2_linear_models\';
if ~exist(outFolder, 'dir')
    mkdir(outFolder);
end

outFile = fullfile(outFolder, 'AR_univariate_global_2019_2021.mat');
checkpointFile = fullfile(outFolder, 'AR_univariate_global_2019_2021_checkpoint.mat');

% If checkpoint file exists, resume unfinished blocks.
% Set false to delete existing checkpoint and restart from block 1.
resumeFromCheckpoint = true;

numWorkers = 15;
block_num = 10;

p_list = [1, 6, 12, 24, 36, 48, 60];
testYears = [2019, 2020, 2021];

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

%% ===================== Initialize or resume valid-grid outputs =====================

nRound = numel(testYears);
nPredMonth = nRound * 12;

block_size = max(1, ceil(nValid / block_num));
total_blocks = ceil(nValid / block_size);

if resumeFromCheckpoint && exist(checkpointFile, 'file')
    fprintf('\nLoading checkpoint:\n%s\n', checkpointFile);

    C = load(checkpointFile);

    AR_Pred_valid = C.AR_Pred_valid;
    AR_NRMSE_valid = C.AR_NRMSE_valid;
    AR_TDA_valid = C.AR_TDA_valid;
    AR_best_p_valid = C.AR_best_p_valid;
    AR_Val_NRMSE_best_valid = C.AR_Val_NRMSE_best_valid;
    blockDone = C.blockDone;

    if isfield(C, 'cumulativeComputeTimeSeconds')
        cumulativeComputeTimeSeconds = C.cumulativeComputeTimeSeconds;
    else
        cumulativeComputeTimeSeconds = 0;
    end

    clear C
else
    AR_Pred_valid = NaN(nValid, nPredMonth);
    AR_NRMSE_valid = NaN(nValid, nRound);
    AR_TDA_valid = NaN(nValid, nRound);
    AR_best_p_valid = NaN(nValid, nRound);
    AR_Val_NRMSE_best_valid = NaN(nValid, nRound);

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

%% ===================== Global AR computation =====================

fprintf('\nStarting AR global computation...\n');

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

        [pred36, nrmse3, tda3, bestP3, valNRMSEBest3] = runAROnePixelAllRounds( ...
            Y_pixel, p_list, testYears);

        blockPred(i, :) = pred36(:)';
        blockNRMSE(i, :) = nrmse3(:)';
        blockTDA(i, :) = tda3(:)';
        blockBestP(i, :) = bestP3(:)';
        blockValNRMSEBest(i, :) = valNRMSEBest3(:)';
    end

    blockElapsed = toc(blockTic);
    cumulativeComputeTimeSeconds = cumulativeComputeTimeSeconds + blockElapsed;

    AR_Pred_valid(currentValidRows, :) = blockPred;
    AR_NRMSE_valid(currentValidRows, :) = blockNRMSE;
    AR_TDA_valid(currentValidRows, :) = blockTDA;
    AR_best_p_valid(currentValidRows, :) = blockBestP;
    AR_Val_NRMSE_best_valid(currentValidRows, :) = blockValNRMSEBest;

    blockDone(block) = true;

    save(checkpointFile, ...
        'AR_Pred_valid', ...
        'AR_NRMSE_valid', ...
        'AR_TDA_valid', ...
        'AR_best_p_valid', ...
        'AR_Val_NRMSE_best_valid', ...
        'blockDone', ...
        'cumulativeComputeTimeSeconds', ...
        'p_list', 'testYears', 'validLinearIndex', ...
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
fprintf('\nAR global computation finished. Compute time this run: %.2f minutes\n', elapsedComputeThisRun / 60);
fprintf('AR cumulative block computation time: %.2f minutes\n', cumulativeComputeTimeSeconds / 60);

%% ===================== Map valid-grid outputs back to global fields =====================

fprintf('\nMapping valid-grid outputs back to global fields...\n');

AR_Pred = mapValidToGrid3D(AR_Pred_valid, validLinearIndex, nLat, nLon);
AR_NRMSE = mapValidToGrid3D(AR_NRMSE_valid, validLinearIndex, nLat, nLon);
AR_TDA = mapValidToGrid3D(AR_TDA_valid, validLinearIndex, nLat, nLon);
AR_best_p = mapValidToGrid3D(AR_best_p_valid, validLinearIndex, nLat, nLon);
AR_Val_NRMSE_best = mapValidToGrid3D(AR_Val_NRMSE_best_valid, validLinearIndex, nLat, nLon);

AR_NRMSE_mean = mean(AR_NRMSE, 3, 'omitnan');
AR_TDA_mean = mean(AR_TDA, 3, 'omitnan');
AR_Val_NRMSE_best_mean = mean(AR_Val_NRMSE_best, 3, 'omitnan');

predYM = makeYMList(2019, 1, 2021, 12);

yearVec = floor(ym / 100);
monthVec = mod(ym, 100);

%% ===================== Basic summary =====================

fprintf('\nSummary over common valid grid cells:\n');
fprintf('AR_NRMSE_mean: mean = %.4f, median = %.4f, min = %.4f, max = %.4f\n', ...
    mean(AR_NRMSE_mean(CommonValid), 'omitnan'), ...
    median(AR_NRMSE_mean(CommonValid), 'omitnan'), ...
    min(AR_NRMSE_mean(CommonValid), [], 'omitnan'), ...
    max(AR_NRMSE_mean(CommonValid), [], 'omitnan'));

fprintf('AR_TDA_mean: mean = %.2f%%, median = %.2f%%, min = %.2f%%, max = %.2f%%\n', ...
    mean(AR_TDA_mean(CommonValid), 'omitnan'), ...
    median(AR_TDA_mean(CommonValid), 'omitnan'), ...
    min(AR_TDA_mean(CommonValid), [], 'omitnan'), ...
    max(AR_TDA_mean(CommonValid), [], 'omitnan'));

%% ===================== Metadata =====================

Meta = struct();
Meta.model = 'AR';
Meta.framework = 'Groundwater-history-only univariate forecasting';
Meta.input = 'dGWS_valid only';
Meta.period = '2004-01 to 2024-12';
Meta.predictionPeriod = '2019-01 to 2021-12';
Meta.arrayConvention = 'Spatial fields are [lat, lon, time]; AR_Pred time dimension is 2019-01 to 2021-12.';
Meta.rollingStrategy = 'Three rounds: train 2004-2017/2018/2019, validate 2018/2019/2020, test 2019/2020/2021.';
Meta.hyperparameter = 'AR order p selected from p_list by validation-year NRMSE.';
Meta.p_list = p_list;
Meta.finalPrediction = 'no-refit-after-validation: AR coefficients are estimated from Train only; Train + Val is used only as observed forecasting history before Test.';
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
Meta.note2 = 'AR_Val_NRMSE_best is retained for traceability of validation-based p selection.';

%% ===================== Save final output =====================

fprintf('\nSaving final output to:\n%s\n', outFile);

save(outFile, ...
    'AR_Pred', ...
    'AR_NRMSE', 'AR_NRMSE_mean', ...
    'AR_TDA', 'AR_TDA_mean', ...
    'AR_best_p', ...
    'AR_Val_NRMSE_best', 'AR_Val_NRMSE_best_mean', ...
    'p_list', 'testYears', 'predYM', ...
    'CommonMask', 'CommonValid', 'validLinearIndex', 'nValid', ...
    'validLat', 'validLon', ...
    'lat', 'lon', 'ym', 'yearVec', 'monthVec', ...
    'Meta', ...
    '-v7.3');

fprintf('Final output saved.\n');

%% ===================== Diagnostic plot =====================

figure;

p = pcolor(lon, lat, AR_NRMSE_mean);
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

title('AR mean NRMSE, testing years 2019-2021');
xlabel('Longitude');
ylabel('Latitude');

set(gca, 'FontName', 'Times New Roman', 'FontSize', 11);

try
    set(p, 'AlphaData', ~isnan(AR_NRMSE_mean));
catch
end

%% ===================== Local functions =====================

function [pred36, nrmse3, tda3, bestP3, valNRMSEBest3] = runAROnePixelAllRounds(Y, p_list, testYears)
    Y = double(Y(:));

    nRound = numel(testYears);

    pred36 = NaN(36, 1);
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

        valNRMSE_all = Inf(numel(p_list), 1);

        for k = 1:numel(p_list)
            p = p_list(k);

            if p >= numel(Y_train)
                continue;
            end

            try
                model = ar(Y_train, p);
                Y_val_pred = forecastNumeric(model, Y_train, 12);
                valNRMSE_all(k) = calcNRMSE(Y_val, Y_val_pred);
            catch
                valNRMSE_all(k) = Inf;
            end
        end

        [valBest, bestIdx] = min(valNRMSE_all);

        if isempty(bestIdx) || ~isfinite(valBest)
            continue;
        end

        bestP = p_list(bestIdx);

        % No-refit-after-validation:
        % keep the AR coefficients estimated from Train only.
        % Use Train + Val only as the observed forecasting history before Test.
        Y_train_val = Y(1:(trainEndIdx + 12));

        try
            finalModel = ar(Y_train, bestP);
            Y_test_pred = forecastNumeric(finalModel, Y_train_val, 12);
        catch
            continue;
        end

        if numel(Y_test_pred) ~= 12 || any(~isfinite(Y_test_pred))
            continue;
        end

        predRange = ((r - 1) * 12 + 1):(r * 12);

        pred36(predRange) = Y_test_pred(:);
        nrmse3(r) = calcNRMSE(Y_test, Y_test_pred);
        tda3(r) = calcTDA(Y_test, Y_test_pred);
        bestP3(r) = bestP;
        valNRMSEBest3(r) = valBest;
    end
end

function yhat = forecastNumeric(model, historyY, horizon)
    yhat = forecast(model, historyY(:), horizon);

    if isa(yhat, 'iddata')
        yhat = yhat.OutputData;

        if iscell(yhat)
            yhat = yhat{1};
        end
    end

    yhat = double(yhat(:));

    if numel(yhat) ~= horizon
        error('Forecast length is %d, expected %d.', numel(yhat), horizon);
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

%% results for 2019-2021

% AR global computation finished. Compute time this run: 173.51 minutes
% AR cumulative block computation time: 173.50 minutes
% 
% Mapping valid-grid outputs back to global fields...
% 
% Summary over common valid grid cells:
% AR_NRMSE_mean: mean = 0.2943, median = 0.2552, min = 0.0663, max = 218.4904
% AR_TDA_mean: mean = 59.63%, median = 57.58%, min = 12.12%, max = 100.00%
