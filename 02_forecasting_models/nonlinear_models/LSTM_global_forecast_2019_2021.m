%% LSTM model, univariate global groundwater forecasting
% Target:
%   dGWS_valid from GLDAS_common_valid_mask_1deg_200401_202412.mat
%
% Framework:
%   Groundwater-history-only, LSTM Seq-to-Vector model
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
% LSTM setting:
%   Seq-to-Vector direct 12-month prediction.
%   sequenceInputLayer(1)
%   lstmLayer(numHiddenUnits, OutputMode='last')
%   fullyConnectedLayer(12)
%   regressionLayer
%
% Standardization:
%   Training-only column-wise standardization for input sequence positions.
%   Training-only column-wise standardization for 12 output horizons.
%
% Forecasting strategy:
%   Seq-to-Vector direct prediction.
%   For each p, one LSTM model directly outputs 12 future months.
%
% Early stopping:
%   Validation-year data are used for validation-based early stopping.
%   Validation-year NRMSE is also used for p selection.
%
% Final prediction strategy:
%   no-refit-after-validation.
%   First train candidate LSTM models using the training period.
%   Then select best p using mean validation-year NRMSE across numRepeat.
%   For the test year, keep the LSTM model estimated from the training period only,
%   and use the observed Train + Val series as the forecasting history.
%   The final test prediction is the monthly mean of numRepeat LSTM predictions.
%
% Parallel strategy:
%   Use 10 blocks and 15 workers.
%   Save checkpoint after each completed block.
%
% Output:
%   LSTM_Pred                    [150, 360, 36]
%   LSTM_NRMSE                   [150, 360, 3]
%   LSTM_NRMSE_mean              [150, 360]
%   LSTM_TDA                     [150, 360, 3]
%   LSTM_TDA_mean                [150, 360]
%   LSTM_best_p                  [150, 360, 3]
%   LSTM_Val_NRMSE_best          [150, 360, 3]
%   LSTM_Val_NRMSE_best_mean     [150, 360]

clc;
clearvars;

%% ===================== User settings =====================

dataFile = 'E:\P1_SKLGB\M_manuscript1\s1_data_results\s1_data_prepare\GLDAS_common_valid_mask_1deg_200401_202412.mat';

outFolder = 'E:\P1_SKLGB\M_manuscript2\s1_data_results\s3_nonlinear_models\';
if ~exist(outFolder, 'dir')
    mkdir(outFolder);
end

% uc1 means uncertainty check
outFile = fullfile(outFolder, 'LSTM_univariate_global_2019_2021.mat');
checkpointFile = fullfile(outFolder, 'LSTM_univariate_global_2019_2021_checkpoint.mat');

% If checkpoint file exists, resume unfinished blocks.
% Set false to delete existing checkpoint and restart from block 1.
resumeFromCheckpoint = true;

numWorkers = 15;
block_num = 10;

p_list = [1, 6, 12, 24, 36, 48, 60];
testYears = [2019, 2020, 2021];

numRepeat = 1;
% switch randomSeed number for uncertainty check
randomSeed = 20260605;
% randomSeed = 20260719;

lstmParam = struct();
lstmParam.numPredict = 12;
lstmParam.numFeatures = 1;
lstmParam.numHiddenUnits = 32;
lstmParam.initialLearnRate = 0.01;
lstmParam.miniBatchSize = 8;
lstmParam.maxEpochs = 500;
lstmParam.validationPatience = 10;
lstmParam.gradientThreshold = 1;
lstmParam.optimizer = 'adam';
lstmParam.executionEnvironment = 'cpu';
lstmParam.shuffle = 'never';

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

fprintf('\nLSTM settings:\n');
fprintf('  numPredict           : %d\n', lstmParam.numPredict);
fprintf('  numFeatures          : %d\n', lstmParam.numFeatures);
fprintf('  numHiddenUnits       : %d\n', lstmParam.numHiddenUnits);
fprintf('  InitialLearnRate     : %.5f\n', lstmParam.initialLearnRate);
fprintf('  MiniBatchSize        : %d\n', lstmParam.miniBatchSize);
fprintf('  MaxEpochs            : %d\n', lstmParam.maxEpochs);
fprintf('  ValidationPatience   : %d\n', lstmParam.validationPatience);
fprintf('  GradientThreshold    : %.4g\n', lstmParam.gradientThreshold);
fprintf('  Optimizer            : %s\n', lstmParam.optimizer);
fprintf('  ExecutionEnvironment : %s\n', lstmParam.executionEnvironment);
fprintf('  Shuffle              : %s\n', lstmParam.shuffle);
fprintf('  numRepeat            : %d\n', numRepeat);
fprintf('  randomSeed           : %d\n', randomSeed);

%% ===================== Initialize or resume valid-grid outputs =====================

nRound = numel(testYears);
horizon = lstmParam.numPredict;
nPredMonth = nRound * horizon;

block_size = max(1, ceil(nValid / block_num));
total_blocks = ceil(nValid / block_size);

if resumeFromCheckpoint && exist(checkpointFile, 'file')
    fprintf('\nLoading checkpoint:\n%s\n', checkpointFile);

    C = load(checkpointFile);

    LSTM_Pred_valid = C.LSTM_Pred_valid;
    LSTM_NRMSE_valid = C.LSTM_NRMSE_valid;
    LSTM_TDA_valid = C.LSTM_TDA_valid;
    LSTM_best_p_valid = C.LSTM_best_p_valid;
    LSTM_Val_NRMSE_best_valid = C.LSTM_Val_NRMSE_best_valid;
    blockDone = C.blockDone;

    if isfield(C, 'cumulativeComputeTimeSeconds')
        cumulativeComputeTimeSeconds = C.cumulativeComputeTimeSeconds;
    else
        cumulativeComputeTimeSeconds = 0;
    end

    clear C
else
    LSTM_Pred_valid = NaN(nValid, nPredMonth);
    LSTM_NRMSE_valid = NaN(nValid, nRound);
    LSTM_TDA_valid = NaN(nValid, nRound);
    LSTM_best_p_valid = NaN(nValid, nRound);
    LSTM_Val_NRMSE_best_valid = NaN(nValid, nRound);

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

%% ===================== Global LSTM computation =====================

fprintf('\nStarting LSTM global computation...\n');

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

        [pred36, nrmse3, tda3, bestP3, valNRMSEBest3] = runLSTMOnePixelAllRounds( ...
            Y_pixel, p_list, testYears, lstmParam, numRepeat, randomSeed, globalRow);

        blockPred(i, :) = pred36(:)';
        blockNRMSE(i, :) = nrmse3(:)';
        blockTDA(i, :) = tda3(:)';
        blockBestP(i, :) = bestP3(:)';
        blockValNRMSEBest(i, :) = valNRMSEBest3(:)';
    end

    blockElapsed = toc(blockTic);
    cumulativeComputeTimeSeconds = cumulativeComputeTimeSeconds + blockElapsed;

    LSTM_Pred_valid(currentValidRows, :) = blockPred;
    LSTM_NRMSE_valid(currentValidRows, :) = blockNRMSE;
    LSTM_TDA_valid(currentValidRows, :) = blockTDA;
    LSTM_best_p_valid(currentValidRows, :) = blockBestP;
    LSTM_Val_NRMSE_best_valid(currentValidRows, :) = blockValNRMSEBest;

    blockDone(block) = true;

    save(checkpointFile, ...
        'LSTM_Pred_valid', ...
        'LSTM_NRMSE_valid', ...
        'LSTM_TDA_valid', ...
        'LSTM_best_p_valid', ...
        'LSTM_Val_NRMSE_best_valid', ...
        'blockDone', ...
        'cumulativeComputeTimeSeconds', ...
        'p_list', 'testYears', ...
        'numRepeat', 'lstmParam', 'randomSeed', ...
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
fprintf('\nLSTM global computation finished. Compute time this run: %.2f minutes\n', elapsedComputeThisRun / 60);
fprintf('LSTM cumulative block computation time: %.2f minutes\n', cumulativeComputeTimeSeconds / 60);

%% ===================== Map valid-grid outputs back to global fields =====================

fprintf('\nMapping valid-grid outputs back to global fields...\n');

LSTM_Pred = mapValidToGrid3D(LSTM_Pred_valid, validLinearIndex, nLat, nLon);
LSTM_NRMSE = mapValidToGrid3D(LSTM_NRMSE_valid, validLinearIndex, nLat, nLon);
LSTM_TDA = mapValidToGrid3D(LSTM_TDA_valid, validLinearIndex, nLat, nLon);
LSTM_best_p = mapValidToGrid3D(LSTM_best_p_valid, validLinearIndex, nLat, nLon);
LSTM_Val_NRMSE_best = mapValidToGrid3D(LSTM_Val_NRMSE_best_valid, validLinearIndex, nLat, nLon);

LSTM_NRMSE_mean = mean(LSTM_NRMSE, 3, 'omitnan');
LSTM_TDA_mean = mean(LSTM_TDA, 3, 'omitnan');
LSTM_Val_NRMSE_best_mean = mean(LSTM_Val_NRMSE_best, 3, 'omitnan');

predYM = makeYMList(2019, 1, 2021, 12);

yearVec = floor(ym / 100);
monthVec = mod(ym, 100);

%% ===================== Basic summary =====================

fprintf('\nSummary over common valid grid cells:\n');
fprintf('LSTM_NRMSE_mean: mean = %.4f, median = %.4f, min = %.4f, max = %.4f\n', ...
    mean(LSTM_NRMSE_mean(CommonValid), 'omitnan'), ...
    median(LSTM_NRMSE_mean(CommonValid), 'omitnan'), ...
    min(LSTM_NRMSE_mean(CommonValid), [], 'omitnan'), ...
    max(LSTM_NRMSE_mean(CommonValid), [], 'omitnan'));

fprintf('LSTM_TDA_mean: mean = %.2f%%, median = %.2f%%, min = %.2f%%, max = %.2f%%\n', ...
    mean(LSTM_TDA_mean(CommonValid), 'omitnan'), ...
    median(LSTM_TDA_mean(CommonValid), 'omitnan'), ...
    min(LSTM_TDA_mean(CommonValid), [], 'omitnan'), ...
    max(LSTM_TDA_mean(CommonValid), [], 'omitnan'));

%% ===================== Metadata =====================

Meta = struct();
Meta.model = 'LSTM';
Meta.framework = 'Groundwater-history-only univariate forecasting';
Meta.input = 'dGWS_valid only';
Meta.period = '2004-01 to 2024-12';
Meta.predictionPeriod = '2019-01 to 2021-12';
Meta.arrayConvention = 'Spatial fields are [lat, lon, time]; LSTM_Pred time dimension is 2019-01 to 2021-12.';
Meta.rollingStrategy = 'Three rounds: train 2004-2017/2018/2019, validate 2018/2019/2020, test 2019/2020/2021.';
Meta.hyperparameter = 'History window length p selected from p_list by validation-year NRMSE.';
Meta.p_list = p_list;
Meta.lstmParam = lstmParam;
Meta.numRepeat = numRepeat;
Meta.randomSeed = randomSeed;
Meta.standardization = 'Training-only column-wise standardization for input sequence positions and 12 output horizons.';
Meta.architecture = 'sequenceInputLayer(1)-lstmLayer(numHiddenUnits,OutputMode=last)-fullyConnectedLayer(12)-regressionLayer.';
Meta.forecasting = 'Seq-to-Vector direct 12-month prediction.';
Meta.earlyStopping = 'Validation-year data are used for validation-based early stopping and p selection.';
Meta.finalPrediction = 'no-refit-after-validation: LSTM models are estimated from Train only; Train + Val is used only as observed forecasting history before Test.';
Meta.repeatStrategy = 'Validation p selection uses mean validation NRMSE across numRepeat LSTM fits. Final test prediction is the monthly mean of numRepeat LSTM predictions.';
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
Meta.note1 = 'Validation-year NRMSE is used for both p selection and early stopping.';
Meta.note2 = 'LSTM_Val_NRMSE_best is retained for traceability of validation-based p selection.';
Meta.note3 = 'numRepeat is retained as a user-adjustable setting; current default is 1 for computational feasibility.';

%% ===================== Save final output =====================

fprintf('\nSaving final output to:\n%s\n', outFile);

save(outFile, ...
    'LSTM_Pred', ...
    'LSTM_NRMSE', 'LSTM_NRMSE_mean', ...
    'LSTM_TDA', 'LSTM_TDA_mean', ...
    'LSTM_best_p', ...
    'LSTM_Val_NRMSE_best', 'LSTM_Val_NRMSE_best_mean', ...
    'p_list', 'testYears', ...
    'numRepeat', 'lstmParam', 'randomSeed', 'predYM', ...
    'CommonMask', 'CommonValid', 'validLinearIndex', 'nValid', ...
    'validLat', 'validLon', ...
    'lat', 'lon', 'ym', 'yearVec', 'monthVec', ...
    'Meta', ...
    '-v7.3');

fprintf('Final output saved.\n');

%% ===================== Diagnostic plot =====================

figure;

p = pcolor(lon, lat, LSTM_NRMSE_mean);
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

title('LSTM mean NRMSE, testing years 2019-2021');
xlabel('Longitude');
ylabel('Latitude');

set(gca, 'FontName', 'Times New Roman', 'FontSize', 11);

try
    set(p, 'AlphaData', ~isnan(LSTM_NRMSE_mean));
catch
end

%% ===================== Local functions =====================

function [pred36, nrmse3, tda3, bestP3, valNRMSEBest3] = runLSTMOnePixelAllRounds( ...
    Y, p_list, testYears, lstmParam, numRepeat, randomSeed, pointID)

    Y = double(Y(:));

    nRound = numel(testYears);
    horizon = lstmParam.numPredict;

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

        trainEndIdx = (testYear - 2005) * 12;

        valIdx = (trainEndIdx + 1):(trainEndIdx + horizon);
        testIdx = (trainEndIdx + horizon + 1):(trainEndIdx + 2 * horizon);

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

            valBlock = [Y_train(end-p+1:end); Y_val(:)];

            for rep = 1:numRepeat
                try
                    seedVal = makeSeed(randomSeed, pointID, r, k, rep, 0);

                    lstmPack = fitLSTMSeq2VectorFromSeries( ...
                        Y_train, p, lstmParam, seedVal, valBlock);

                    initWindowVal = Y_train(end-p+1:end);
                    Y_val_pred = predictLSTMSeq2Vector(lstmPack, initWindowVal);

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
        % keep the LSTM model estimated from Train only.
        % Validation year is used only for early stopping and p selection.
        % Train + Val is used only as observed forecasting history before Test.
        Y_train_val = Y(1:(trainEndIdx + horizon));

        % Rebuild validation block using bestP, not the last p from the search loop.
        valBlockBest = [Y_train(end-bestP+1:end); Y_val(:)];

        Y_test_pred_repeat = NaN(horizon, numRepeat);

        for rep = 1:numRepeat
            try
                seedTest = makeSeed(randomSeed, pointID, r, bestIdx, rep, 1);

                finalPack = fitLSTMSeq2VectorFromSeries( ...
                    Y_train, bestP, lstmParam, seedTest, valBlockBest);

                initWindowTest = Y_train_val(end-bestP+1:end);
                Y_test_pred_repeat(:, rep) = predictLSTMSeq2Vector(finalPack, initWindowTest);
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

function lstmPack = fitLSTMSeq2VectorFromSeries(Y_series, p, lstmParam, seedVal, valBlock)
    Y_series = double(Y_series(:));
    numPredict = lstmParam.numPredict;

    if numel(Y_series) <= p + numPredict - 1
        error('Training series is too short for p = %d and numPredict = %d.', p, numPredict);
    end

    [InputTrain, OutputTrain] = buildSeq2VectorSamples(Y_series, p, numPredict);

    [InputTrainNorm, mu_X, sigma_X] = standardizeTrainX(InputTrain);
    [OutputTrainNorm, mu_Y, sigma_Y] = standardizeTrainY(OutputTrain);

    nTrain = size(InputTrainNorm, 1);

    InputTrainCell = cell(nTrain, 1);
    for j = 1:nTrain
        InputTrainCell{j} = InputTrainNorm(j, :);
    end

    OutputTrainMatrix = OutputTrainNorm;

    layers = [
        sequenceInputLayer(lstmParam.numFeatures, 'Name', 'input')
        lstmLayer(lstmParam.numHiddenUnits, 'OutputMode', 'last', 'Name', 'lstm')
        fullyConnectedLayer(numPredict, 'Name', 'fc')
        regressionLayer('Name', 'regression')];

    validationFrequency = max(1, floor(nTrain / lstmParam.miniBatchSize));

    seedSafe = mod(seedVal, 2^32 - 1);
    if seedSafe == 0
        seedSafe = 1;
    end
    rng(seedSafe, 'twister');

    if ~isempty(valBlock)
        valBlock = double(valBlock(:));

        if numel(valBlock) ~= p + numPredict
            error('Validation block length is %d, expected p + numPredict = %d.', ...
                numel(valBlock), p + numPredict);
        end

        InputVal = valBlock(1:p)';
        OutputVal = valBlock(p+1:end)';

        InputValNorm = (InputVal - mu_X) ./ sigma_X;
        OutputValNorm = (OutputVal - mu_Y) ./ sigma_Y;

        InputValCell = {InputValNorm};
        OutputValMatrix = OutputValNorm;

        options = trainingOptions(lstmParam.optimizer, ...
            'ExecutionEnvironment', lstmParam.executionEnvironment, ...
            'MaxEpochs', lstmParam.maxEpochs, ...
            'MiniBatchSize', lstmParam.miniBatchSize, ...
            'InitialLearnRate', lstmParam.initialLearnRate, ...
            'ValidationData', {InputValCell, OutputValMatrix}, ...
            'ValidationFrequency', validationFrequency, ...
            'ValidationPatience', lstmParam.validationPatience, ...
            'Verbose', 0, ...
            'Plots', 'none', ...
            'Shuffle', lstmParam.shuffle, ...
            'GradientThreshold', lstmParam.gradientThreshold);
    else
        options = trainingOptions(lstmParam.optimizer, ...
            'ExecutionEnvironment', lstmParam.executionEnvironment, ...
            'MaxEpochs', lstmParam.maxEpochs, ...
            'MiniBatchSize', lstmParam.miniBatchSize, ...
            'InitialLearnRate', lstmParam.initialLearnRate, ...
            'Verbose', 0, ...
            'Plots', 'none', ...
            'Shuffle', lstmParam.shuffle, ...
            'GradientThreshold', lstmParam.gradientThreshold);
    end

    Mdl = trainNetwork(InputTrainCell, OutputTrainMatrix, layers, options);

    lstmPack = struct();
    lstmPack.Mdl = Mdl;
    lstmPack.p = p;
    lstmPack.numPredict = numPredict;
    lstmPack.mu_X = mu_X;
    lstmPack.sigma_X = sigma_X;
    lstmPack.mu_Y = mu_Y;
    lstmPack.sigma_Y = sigma_Y;
    lstmPack.lstmParam = lstmParam;
end

function [X, Y_target] = buildSeq2VectorSamples(Y_series, p, numPredict)
    Y_series = double(Y_series(:));

    nSample = numel(Y_series) - p - numPredict + 1;

    if nSample <= 0
        error('No Seq-to-Vector samples can be constructed.');
    end

    X = zeros(nSample, p);
    Y_target = zeros(nSample, numPredict);

    for j = 1:nSample
        X(j, :) = Y_series(j:j+p-1)';
        Y_target(j, :) = Y_series(j+p:j+p+numPredict-1)';
    end
end

function Y_pred = predictLSTMSeq2Vector(lstmPack, initialWindow)
    Mdl = lstmPack.Mdl;
    p = lstmPack.p;

    mu_X = lstmPack.mu_X;
    sigma_X = lstmPack.sigma_X;
    mu_Y = lstmPack.mu_Y;
    sigma_Y = lstmPack.sigma_Y;

    initialWindow = double(initialWindow(:)');

    if numel(initialWindow) ~= p
        error('Initial window length is %d, expected p = %d.', numel(initialWindow), p);
    end

    inputNorm = (initialWindow - mu_X) ./ sigma_X;

    yPredNorm = predict(Mdl, inputNorm);
    yPredNorm = double(yPredNorm(:)');

    Y_pred = yPredNorm .* sigma_Y + mu_Y;
    Y_pred = double(Y_pred(:));
end

function [X_norm, mu_X, sigma_X] = standardizeTrainX(X)
    X = double(X);

    mu_X = mean(X, 1);
    sigma_X = std(X, 0, 1);

    sigma_X(~isfinite(sigma_X) | sigma_X == 0) = 1;

    X_norm = (X - mu_X) ./ sigma_X;
end

function [Y_norm, mu_Y, sigma_Y] = standardizeTrainY(Y)
    Y = double(Y);

    mu_Y = mean(Y, 1);
    sigma_Y = std(Y, 0, 1);

    sigma_Y(~isfinite(sigma_Y) | sigma_Y == 0) = 1;

    Y_norm = (Y - mu_Y) ./ sigma_Y;
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