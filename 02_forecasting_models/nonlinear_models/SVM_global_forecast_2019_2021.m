%% SVM model, univariate global groundwater forecasting
% Target:
%   dGWS_valid from GLDAS_common_valid_mask_1deg_200401_202412.mat
%
% Framework:
%   Groundwater-history-only, SVM model
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
%   box_list = [1, 2, 5, 10, 20, 50, 100]
%
% Fixed SVM setting:
%   fitrsvm with RBF kernel.
%   KernelScale = 'auto'
%   Epsilon = 0.1
%   BoxConstraint is selected by validation-year NRMSE.
%
% Standardization:
%   Training-only column-wise standardization for input features.
%   Training-only standardization for target variable.
%
% Forecasting strategy:
%   Horizon-specific direct prediction.
%   For each p and BoxConstraint, train 12 SVM models for horizons h = 1, ..., 12.
%
% Final prediction strategy:
%   no-refit-after-validation.
%   First train candidate SVM models using the training period.
%   Then select best p and best BoxConstraint using validation-year NRMSE.
%   For the test year, keep the SVM models estimated from the training period only,
%   and use the observed Train + Val series as the forecasting history.
%
% Parallel strategy:
%   Use 10 blocks and 15 workers.
%   Save checkpoint after each completed block.
%
% Output:
%   SVM_Pred                    [150, 360, 36]
%   SVM_NRMSE                   [150, 360, 3]
%   SVM_NRMSE_mean              [150, 360]
%   SVM_TDA                     [150, 360, 3]
%   SVM_TDA_mean                [150, 360]
%   SVM_best_p                  [150, 360, 3]
%   SVM_best_box                [150, 360, 3]
%   SVM_Val_NRMSE_best          [150, 360, 3]
%   SVM_Val_NRMSE_best_mean     [150, 360]

clc;
clearvars;

%% ===================== User settings =====================

dataFile = 'E:\P1_SKLGB\M_manuscript1\s1_data_results\s1_data_prepare\GLDAS_common_valid_mask_1deg_200401_202412.mat';

outFolder = 'E:\P1_SKLGB\M_manuscript2\s1_data_results\s3_nonlinear_models\';
if ~exist(outFolder, 'dir')
    mkdir(outFolder);
end

% uc1 means uncertainty check
outFile = fullfile(outFolder, 'SVM_univariate_global_2019_2021.mat');
checkpointFile = fullfile(outFolder, 'SVM_univariate_global_2019_2021_checkpoint.mat');

% If checkpoint file exists, resume unfinished blocks.
% Set false to delete existing checkpoint and restart from block 1.
resumeFromCheckpoint = true;

numWorkers = 15;
block_num = 10;

p_list = [1, 6, 12, 24, 36, 48, 60];
box_list = [1, 2, 5, 10, 20, 50, 100];

testYears = [2019, 2020, 2021];
horizon = 12;

% switch randomSeed number for uncertainty check
randomSeed = 20260605;
% randomSeed = 20260719;

svmParam = struct();
svmParam.KernelFunction = 'rbf';
svmParam.KernelScale = 'auto';
svmParam.Epsilon = 0.1;

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

fprintf('\nSVM settings:\n');
fprintf('  KernelFunction : %s\n', svmParam.KernelFunction);
fprintf('  KernelScale    : %s\n', svmParam.KernelScale);
fprintf('  Epsilon        : %.4g\n', svmParam.Epsilon);
fprintf('  box_list       : %s\n', mat2str(box_list));
fprintf('  randomSeed     : %d\n', randomSeed);

%% ===================== Initialize or resume valid-grid outputs =====================

nRound = numel(testYears);
nPredMonth = nRound * horizon;

block_size = max(1, ceil(nValid / block_num));
total_blocks = ceil(nValid / block_size);

if resumeFromCheckpoint && exist(checkpointFile, 'file')
    fprintf('\nLoading checkpoint:\n%s\n', checkpointFile);

    C = load(checkpointFile);

    SVM_Pred_valid = C.SVM_Pred_valid;
    SVM_NRMSE_valid = C.SVM_NRMSE_valid;
    SVM_TDA_valid = C.SVM_TDA_valid;
    SVM_best_p_valid = C.SVM_best_p_valid;
    SVM_best_box_valid = C.SVM_best_box_valid;
    SVM_Val_NRMSE_best_valid = C.SVM_Val_NRMSE_best_valid;
    blockDone = C.blockDone;

    if isfield(C, 'cumulativeComputeTimeSeconds')
        cumulativeComputeTimeSeconds = C.cumulativeComputeTimeSeconds;
    else
        cumulativeComputeTimeSeconds = 0;
    end

    clear C
else
    SVM_Pred_valid = NaN(nValid, nPredMonth);
    SVM_NRMSE_valid = NaN(nValid, nRound);
    SVM_TDA_valid = NaN(nValid, nRound);
    SVM_best_p_valid = NaN(nValid, nRound);
    SVM_best_box_valid = NaN(nValid, nRound);
    SVM_Val_NRMSE_best_valid = NaN(nValid, nRound);

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

%% ===================== Global SVM computation =====================

fprintf('\nStarting SVM global computation...\n');

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
    blockBestBox = NaN(block_len, nRound);
    blockValNRMSEBest = NaN(block_len, nRound);

    blockTic = tic;

    parfor i = 1:block_len
        globalRow = currentValidRows(i);

        Y_pixel = Y_valid_const.Value(globalRow, :)';

        [pred36, nrmse3, tda3, bestP3, bestBox3, valNRMSEBest3] = runSVMOnePixelAllRounds( ...
            Y_pixel, p_list, box_list, testYears, horizon, svmParam, randomSeed, globalRow);

        blockPred(i, :) = pred36(:)';
        blockNRMSE(i, :) = nrmse3(:)';
        blockTDA(i, :) = tda3(:)';
        blockBestP(i, :) = bestP3(:)';
        blockBestBox(i, :) = bestBox3(:)';
        blockValNRMSEBest(i, :) = valNRMSEBest3(:)';
    end

    blockElapsed = toc(blockTic);
    cumulativeComputeTimeSeconds = cumulativeComputeTimeSeconds + blockElapsed;

    SVM_Pred_valid(currentValidRows, :) = blockPred;
    SVM_NRMSE_valid(currentValidRows, :) = blockNRMSE;
    SVM_TDA_valid(currentValidRows, :) = blockTDA;
    SVM_best_p_valid(currentValidRows, :) = blockBestP;
    SVM_best_box_valid(currentValidRows, :) = blockBestBox;
    SVM_Val_NRMSE_best_valid(currentValidRows, :) = blockValNRMSEBest;

    blockDone(block) = true;

    save(checkpointFile, ...
        'SVM_Pred_valid', ...
        'SVM_NRMSE_valid', ...
        'SVM_TDA_valid', ...
        'SVM_best_p_valid', ...
        'SVM_best_box_valid', ...
        'SVM_Val_NRMSE_best_valid', ...
        'blockDone', ...
        'cumulativeComputeTimeSeconds', ...
        'p_list', 'box_list', 'testYears', 'horizon', ...
        'svmParam', 'randomSeed', 'validLinearIndex', ...
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
fprintf('\nSVM global computation finished. Compute time this run: %.2f minutes\n', elapsedComputeThisRun / 60);
fprintf('SVM cumulative block computation time: %.2f minutes\n', cumulativeComputeTimeSeconds / 60);

%% ===================== Map valid-grid outputs back to global fields =====================

fprintf('\nMapping valid-grid outputs back to global fields...\n');

SVM_Pred = mapValidToGrid3D(SVM_Pred_valid, validLinearIndex, nLat, nLon);
SVM_NRMSE = mapValidToGrid3D(SVM_NRMSE_valid, validLinearIndex, nLat, nLon);
SVM_TDA = mapValidToGrid3D(SVM_TDA_valid, validLinearIndex, nLat, nLon);
SVM_best_p = mapValidToGrid3D(SVM_best_p_valid, validLinearIndex, nLat, nLon);
SVM_best_box = mapValidToGrid3D(SVM_best_box_valid, validLinearIndex, nLat, nLon);
SVM_Val_NRMSE_best = mapValidToGrid3D(SVM_Val_NRMSE_best_valid, validLinearIndex, nLat, nLon);

SVM_NRMSE_mean = mean(SVM_NRMSE, 3, 'omitnan');
SVM_TDA_mean = mean(SVM_TDA, 3, 'omitnan');
SVM_Val_NRMSE_best_mean = mean(SVM_Val_NRMSE_best, 3, 'omitnan');

predYM = makeYMList(2019, 1, 2021, 12);

yearVec = floor(ym / 100);
monthVec = mod(ym, 100);

%% ===================== Basic summary =====================

fprintf('\nSummary over common valid grid cells:\n');
fprintf('SVM_NRMSE_mean: mean = %.4f, median = %.4f, min = %.4f, max = %.4f\n', ...
    mean(SVM_NRMSE_mean(CommonValid), 'omitnan'), ...
    median(SVM_NRMSE_mean(CommonValid), 'omitnan'), ...
    min(SVM_NRMSE_mean(CommonValid), [], 'omitnan'), ...
    max(SVM_NRMSE_mean(CommonValid), [], 'omitnan'));

fprintf('SVM_TDA_mean: mean = %.2f%%, median = %.2f%%, min = %.2f%%, max = %.2f%%\n', ...
    mean(SVM_TDA_mean(CommonValid), 'omitnan'), ...
    median(SVM_TDA_mean(CommonValid), 'omitnan'), ...
    min(SVM_TDA_mean(CommonValid), [], 'omitnan'), ...
    max(SVM_TDA_mean(CommonValid), [], 'omitnan'));

%% ===================== Metadata =====================

Meta = struct();
Meta.model = 'SVM';
Meta.framework = 'Groundwater-history-only univariate forecasting';
Meta.input = 'dGWS_valid only';
Meta.period = '2004-01 to 2024-12';
Meta.predictionPeriod = '2019-01 to 2021-12';
Meta.arrayConvention = 'Spatial fields are [lat, lon, time]; SVM_Pred time dimension is 2019-01 to 2021-12.';
Meta.rollingStrategy = 'Three rounds: train 2004-2017/2018/2019, validate 2018/2019/2020, test 2019/2020/2021.';
Meta.hyperparameter = 'History window length p and BoxConstraint selected from p_list and box_list by validation-year NRMSE.';
Meta.p_list = p_list;
Meta.box_list = box_list;
Meta.svmParam = svmParam;
Meta.randomSeed = randomSeed;
Meta.standardization = 'Training-only column-wise standardization for input features; training-only standardization for target variable.';
Meta.forecasting = 'Horizon-specific direct prediction with 12 SVM models for horizons h = 1,...,12.';
Meta.finalPrediction = 'no-refit-after-validation: SVM models are estimated from Train only; Train + Val is used only as observed forecasting history before Test.';
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
Meta.note1 = 'Validation-year NRMSE is used only for p and BoxConstraint selection.';
Meta.note2 = 'SVM_Val_NRMSE_best is retained for traceability of validation-based p and BoxConstraint selection.';

%% ===================== Save final output =====================

fprintf('\nSaving final output to:\n%s\n', outFile);

save(outFile, ...
    'SVM_Pred', ...
    'SVM_NRMSE', 'SVM_NRMSE_mean', ...
    'SVM_TDA', 'SVM_TDA_mean', ...
    'SVM_best_p', 'SVM_best_box', ...
    'SVM_Val_NRMSE_best', 'SVM_Val_NRMSE_best_mean', ...
    'p_list', 'box_list', 'testYears', 'horizon', ...
    'svmParam', 'randomSeed', 'predYM', ...
    'CommonMask', 'CommonValid', 'validLinearIndex', 'nValid', ...
    'validLat', 'validLon', ...
    'lat', 'lon', 'ym', 'yearVec', 'monthVec', ...
    'Meta', ...
    '-v7.3');

fprintf('Final output saved.\n');

%% ===================== Diagnostic plot =====================

figure;

p = pcolor(lon, lat, SVM_NRMSE_mean);
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

title('SVM mean NRMSE, testing years 2019-2021');
xlabel('Longitude');
ylabel('Latitude');

set(gca, 'FontName', 'Times New Roman', 'FontSize', 11);

try
    set(p, 'AlphaData', ~isnan(SVM_NRMSE_mean));
catch
end

%% ===================== Local functions =====================

function [pred36, nrmse3, tda3, bestP3, bestBox3, valNRMSEBest3] = runSVMOnePixelAllRounds( ...
    Y, p_list, box_list, testYears, horizon, svmParam, randomSeed, pointID)

    Y = double(Y(:));

    nRound = numel(testYears);

    pred36 = NaN(nRound * horizon, 1);
    nrmse3 = NaN(nRound, 1);
    tda3 = NaN(nRound, 1);
    bestP3 = NaN(nRound, 1);
    bestBox3 = NaN(nRound, 1);
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

        valNRMSE_all = Inf(numel(p_list), numel(box_list));

        for k = 1:numel(p_list)
            p = p_list(k);

            if numel(Y_train) <= p + horizon - 1
                continue;
            end

            for b = 1:numel(box_list)
                boxVal = box_list(b);

                try
                    seedVal = makeSeed(randomSeed, pointID, r, k, b, 0);
                    svmPack = fitSVMHorizonSpecificFromSeries( ...
                        Y_train, p, horizon, svmParam, boxVal, seedVal);

                    initWindowVal = Y_train(end-p+1:end);
                    Y_val_pred = predictSVMHorizonSpecific(svmPack, initWindowVal);
                    valNRMSE_all(k, b) = calcNRMSE(Y_val, Y_val_pred);
                catch
                    valNRMSE_all(k, b) = Inf;
                end
            end
        end

        [valBest, linearIdx] = min(valNRMSE_all(:));

        if isempty(linearIdx) || ~isfinite(valBest)
            continue;
        end

        [bestPIdx, bestBoxIdx] = ind2sub(size(valNRMSE_all), linearIdx);

        bestP = p_list(bestPIdx);
        bestBox = box_list(bestBoxIdx);

        % No-refit-after-validation:
        % keep the SVM models estimated from Train only.
        % Use Train + Val only as the observed forecasting history before Test.
        Y_train_val = Y(1:(trainEndIdx + 12));

        try
            seedTest = makeSeed(randomSeed, pointID, r, bestPIdx, bestBoxIdx, 1);
            finalPack = fitSVMHorizonSpecificFromSeries( ...
                Y_train, bestP, horizon, svmParam, bestBox, seedTest);

            initWindowTest = Y_train_val(end-bestP+1:end);
            Y_test_pred = predictSVMHorizonSpecific(finalPack, initWindowTest);
        catch
            continue;
        end

        if numel(Y_test_pred) ~= horizon || any(~isfinite(Y_test_pred))
            continue;
        end

        predRange = ((r - 1) * horizon + 1):(r * horizon);

        pred36(predRange) = Y_test_pred(:);
        nrmse3(r) = calcNRMSE(Y_test, Y_test_pred);
        tda3(r) = calcTDA(Y_test, Y_test_pred);
        bestP3(r) = bestP;
        bestBox3(r) = bestBox;
        valNRMSEBest3(r) = valBest;
    end
end

function svmPack = fitSVMHorizonSpecificFromSeries( ...
    Y_series, p, horizon, svmParam, boxVal, baseSeed)

    Y_series = double(Y_series(:));

    modelCell = cell(horizon, 1);
    muXCell = cell(horizon, 1);
    sigmaXCell = cell(horizon, 1);
    muY = NaN(horizon, 1);
    sigmaY = NaN(horizon, 1);

    for h = 1:horizon
        [X, Y_target] = buildLaggedSamplesDirect(Y_series, p, h);

        [X_norm, mu_X, sigma_X] = standardizeTrainX(X);
        [Y_norm, mu_y, sigma_y] = standardizeTrainY(Y_target);

        rng(baseSeed + h, 'twister');

        Mdl = fitrsvm( ...
            X_norm, Y_norm, ...
            'KernelFunction', svmParam.KernelFunction, ...
            'BoxConstraint', boxVal, ...
            'KernelScale', svmParam.KernelScale, ...
            'Epsilon', svmParam.Epsilon, ...
            'Standardize', false, ...
            'Verbose', 0);

        modelCell{h} = Mdl;
        muXCell{h} = mu_X;
        sigmaXCell{h} = sigma_X;
        muY(h) = mu_y;
        sigmaY(h) = sigma_y;
    end

    svmPack = struct();
    svmPack.modelCell = modelCell;
    svmPack.p = p;
    svmPack.horizon = horizon;
    svmPack.muXCell = muXCell;
    svmPack.sigmaXCell = sigmaXCell;
    svmPack.muY = muY;
    svmPack.sigmaY = sigmaY;
    svmPack.svmParam = svmParam;
    svmPack.BoxConstraint = boxVal;
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

function Y_pred = predictSVMHorizonSpecific(svmPack, initialWindow)
    p = svmPack.p;
    horizon = svmPack.horizon;

    initialWindow = double(initialWindow(:)');

    if numel(initialWindow) ~= p
        error('Initial window length is %d, expected p = %d.', numel(initialWindow), p);
    end

    Y_pred = NaN(horizon, 1);

    for h = 1:horizon
        Mdl = svmPack.modelCell{h};
        mu_X = svmPack.muXCell{h};
        sigma_X = svmPack.sigmaXCell{h};
        mu_y = svmPack.muY(h);
        sigma_y = svmPack.sigmaY(h);

        X_new = initialWindow;
        X_new_norm = (X_new - mu_X) ./ sigma_X;

        y_norm = predict(Mdl, X_new_norm);
        y_norm = double(y_norm(1));

        Y_pred(h) = y_norm * sigma_y + mu_y;
    end
end

function [X_norm, mu_X, sigma_X] = standardizeTrainX(X)
    X = double(X);

    mu_X = mean(X, 1);
    sigma_X = std(X, 0, 1);

    sigma_X(~isfinite(sigma_X) | sigma_X == 0) = 1;

    X_norm = (X - mu_X) ./ sigma_X;
end

function [Y_norm, mu_Y, sigma_Y] = standardizeTrainY(Y)
    Y = double(Y(:));

    mu_Y = mean(Y);
    sigma_Y = std(Y, 0);

    if ~isfinite(sigma_Y) || sigma_Y == 0
        sigma_Y = 1;
    end

    Y_norm = (Y - mu_Y) ./ sigma_Y;
end

function seedVal = makeSeed(randomSeed, pointID, roundID, pID, boxID, stageID)
    seedRaw = randomSeed ...
        + pointID * 100 ...
        + roundID * 10000 ...
        + pID * 1000 ...
        + boxID * 10 ...
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