%% Step 1: Build 2019-2024 pixel-year forecast-performance database
% File name:
%   s1_master_database_pixel_year.m
%
% Purpose:
%   Combine 2019-2021 pseudo-forecast results and 2022-2024 original
%   forecast results from 9 groundwater forecasting models.
%
% Basic sampling unit:
%   pixel x forecast_year
%
% Case order:
%   forecast_year -> pixel_id
%
% Forecast years:
%   2019, 2020, 2021, 2022, 2023, 2024
%
% Models:
%   AR, ARIMA, SARIMA, SVM, RF, LSTM, SVMX, RFX, LSTMX
%
% Main annual performance metric:
%
%                   RMSE over future 12 months
%   NRMSE_Q10Q90 = ----------------------------
%                   Q90(pre-origin dGWS)
%                 - Q10(pre-origin dGWS)
%
% Important:
%   For forecast year Y, Q10/Q90 are calculated ONLY from dGWS data
%   available up to December of Y-1.
%
% Output:
%   E:\P1_SKLGB\M_manuscript2\s3_analysis\
%   s1_master_database_pixel_year.mat
%
% Main saved variable:
%   Master
%
% Main dimensions:
%   nValid  = 17,284
%   nYear   = 6
%   nSample = nValid * nYear = 103,704
%
% Master fields:
%   pixel_id               [nSample x 1]
%   lat                    [nSample x 1]
%   lon                    [nSample x 1]
%   forecast_year          [nSample x 1]
%   forecast_origin_ym     [nSample x 1]
%
%   observation            [nSample x 12]
%   prediction             [nSample x 12 x 9]
%
%   preorigin_q10          [nSample x 1]
%   preorigin_q90          [nSample x 1]
%   preorigin_iqr80        [nSample x 1] = Q90-Q10
%
%   rmse                   [nSample x 9]
%   nrmse_q10q90           [nSample x 9]
%
%   model_names
%   spatial lookup information
%   source_files
%   Meta
%
% Missing-data rule:
%   Annual RMSE is calculated only when all 12 observations and all
%   12 predictions for that model are finite. Otherwise RMSE = NaN.
%
%   Q10/Q90 use all finite pre-origin dGWS observations. If fewer than
%   12 finite historical months exist, the scale is set to NaN.

clc;
clearvars;

%% ===================== User settings =====================

analysisFolder = 'E:\P1_SKLGB\M_manuscript2\s3_analysis\';

if ~exist(analysisFolder, 'dir')
    mkdir(analysisFolder);
end

outFile = fullfile( ...
    analysisFolder, ...
    's1_master_database_pixel_year.mat');

rawDataFile = ...
    'E:\P1_SKLGB\M_manuscript1\s1_data_results\s1_data_prepare\GLDAS_common_valid_mask_1deg_200401_202412.mat';

% 2019-2021 model-result folders
p2Base = 'E:\P1_SKLGB\M_manuscript2\s1_data_results\';
p2LinearFolder = fullfile(p2Base, 's2_linear_models');
p2NonlinearFolder = fullfile(p2Base, 's3_nonlinear_models');
p2XFolder = fullfile(p2Base, 's4_nonlinear_x_models');

% 2022-2024 model-result folders
p1Base = 'E:\P1_SKLGB\M_manuscript1\s1_data_results\';
p1LinearFolder = fullfile(p1Base, 's2_linear_models');
p1NonlinearFolder = fullfile(p1Base, 's3_nonlinear_models');
p1XFolder = fullfile(p1Base, 's4_nonlinear_x_models');

forecastYears = 2019:2024;
horizon = 12;

modelNames = { ...
    'AR', 'ARIMA', 'SARIMA', ...
    'SVM', 'RF', 'LSTM', ...
    'SVMX', 'RFX', 'LSTMX'};

nModel = numel(modelNames);
nYear = numel(forecastYears);

%% ===================== Define source files =====================

sourceFiles = repmat(struct( ...
    'model', '', ...
    'predVar', '', ...
    'earlyFile', '', ...
    'lateFile', ''), nModel, 1);

% ----- Linear models -----

sourceFiles(1).model = 'AR';
sourceFiles(1).predVar = 'AR_Pred';
sourceFiles(1).earlyFile = fullfile( ...
    p2LinearFolder, ...
    'AR_univariate_global_2019_2021.mat');
sourceFiles(1).lateFile = fullfile( ...
    p1LinearFolder, ...
    'AR_univariate_global_200401_202412.mat');

sourceFiles(2).model = 'ARIMA';
sourceFiles(2).predVar = 'ARIMA_Pred';
sourceFiles(2).earlyFile = fullfile( ...
    p2LinearFolder, ...
    'ARIMA_univariate_global_2019_2021.mat');
sourceFiles(2).lateFile = fullfile( ...
    p1LinearFolder, ...
    'ARIMA_univariate_global_200401_202412_d0.mat');

sourceFiles(3).model = 'SARIMA';
sourceFiles(3).predVar = 'SARIMA_Pred';
sourceFiles(3).earlyFile = fullfile( ...
    p2LinearFolder, ...
    'SARIMA_univariate_global_2019_2021.mat');
sourceFiles(3).lateFile = fullfile( ...
    p1LinearFolder, ...
    'SARIMA_univariate_global_200401_202412_d0.mat');

% ----- Nonlinear dGWS-only models -----

sourceFiles(4).model = 'SVM';
sourceFiles(4).predVar = 'SVM_Pred';
sourceFiles(4).earlyFile = fullfile( ...
    p2NonlinearFolder, ...
    'SVM_univariate_global_2019_2021.mat');
sourceFiles(4).lateFile = fullfile( ...
    p1NonlinearFolder, ...
    'SVM_univariate_global_200401_202412.mat');

sourceFiles(5).model = 'RF';
sourceFiles(5).predVar = 'RF_Pred';
sourceFiles(5).earlyFile = fullfile( ...
    p2NonlinearFolder, ...
    'RF_univariate_global_2019_2021.mat');
sourceFiles(5).lateFile = fullfile( ...
    p1NonlinearFolder, ...
    'RF_univariate_global_200401_202412.mat');

sourceFiles(6).model = 'LSTM';
sourceFiles(6).predVar = 'LSTM_Pred';
sourceFiles(6).earlyFile = fullfile( ...
    p2NonlinearFolder, ...
    'LSTM_univariate_global_2019_2021.mat');
sourceFiles(6).lateFile = fullfile( ...
    p1NonlinearFolder, ...
    'LSTM_univariate_global_200401_202412.mat');

% ----- Nonlinear + hydroclimatic models -----

sourceFiles(7).model = 'SVMX';
sourceFiles(7).predVar = 'SVMX_Pred';
sourceFiles(7).earlyFile = fullfile( ...
    p2XFolder, ...
    'SVMX_multivariate_global_2019_2021.mat');
sourceFiles(7).lateFile = fullfile( ...
    p1XFolder, ...
    'SVMX_multivariate_global_200401_202412.mat');

sourceFiles(8).model = 'RFX';
sourceFiles(8).predVar = 'RFX_Pred';
sourceFiles(8).earlyFile = fullfile( ...
    p2XFolder, ...
    'RFX_multivariate_global_2019_2021.mat');
sourceFiles(8).lateFile = fullfile( ...
    p1XFolder, ...
    'RFX_multivariate_global_200401_202412.mat');

sourceFiles(9).model = 'LSTMX';
sourceFiles(9).predVar = 'LSTMX_Pred';
sourceFiles(9).earlyFile = fullfile( ...
    p2XFolder, ...
    'LSTMX_multivariate_global_2019_2021.mat');
sourceFiles(9).lateFile = fullfile( ...
    p1XFolder, ...
    'LSTMX_multivariate_global_200401_202412.mat');

%% ===================== Check source files =====================

fprintf('Checking source files...\n');

assertFileExists(rawDataFile, 'Raw dGWS data file');

for m = 1:nModel

    assertFileExists( ...
        sourceFiles(m).earlyFile, ...
        sprintf('%s 2019-2021 result', ...
        sourceFiles(m).model));

    assertFileExists( ...
        sourceFiles(m).lateFile, ...
        sprintf('%s 2022-2024 result', ...
        sourceFiles(m).model));

end

fprintf('All required source files were found.\n');

%% ===================== Load raw dGWS data =====================

fprintf('\nLoading raw dGWS/reference data...\n');

D = load( ...
    rawDataFile, ...
    'dGWS_valid', ...
    'CommonValid', ...
    'validLinearIndex', ...
    'validLat', ...
    'validLon', ...
    'lat', ...
    'lon', ...
    'ym');

requiredRawFields = { ...
    'dGWS_valid', ...
    'CommonValid', ...
    'validLinearIndex', ...
    'validLat', ...
    'validLon', ...
    'lat', ...
    'lon', ...
    'ym'};

for k = 1:numel(requiredRawFields)

    if ~isfield(D, requiredRawFields{k})

        error( ...
            'Raw data file is missing required variable: %s', ...
            requiredRawFields{k});

    end

end

dGWS_valid = double(D.dGWS_valid);

CommonValid = logical(D.CommonValid);

validLinearIndex = double(D.validLinearIndex(:));
validLat = double(D.validLat(:));
validLon = double(D.validLon(:));

latGrid = double(D.lat(:));
lonGrid = double(D.lon(:));

ymRaw = double(D.ym(:));

clear D

[nValid, nRawMonth] = size(dGWS_valid);

nLat = numel(latGrid);
nLon = numel(lonGrid);

%% ===================== Raw-data checks =====================

if numel(validLinearIndex) ~= nValid

    error( ...
        ['Raw data inconsistency: size(dGWS_valid,1) = %d, ', ...
         'but numel(validLinearIndex) = %d.'], ...
        nValid, numel(validLinearIndex));

end

if numel(validLat) ~= nValid || ...
        numel(validLon) ~= nValid

    error( ...
        ['Raw data inconsistency: validLat/validLon ', ...
         'length does not equal nValid.']);

end

if numel(ymRaw) ~= nRawMonth

    error( ...
        ['Raw data inconsistency: size(dGWS_valid,2) = %d, ', ...
         'but numel(ym) = %d.'], ...
        nRawMonth, numel(ymRaw));

end

if ~isequal(size(CommonValid), [nLat, nLon])

    error( ...
        'CommonValid dimensions do not match lat/lon grid.');

end

fprintf('Valid grid cells : %d\n', nValid);
fprintf('Raw months       : %d\n', nRawMonth);
fprintf('Raw period       : %d - %d\n', ymRaw(1), ymRaw(end));
fprintf('Grid size        : %d x %d\n', nLat, nLon);

%% ===================== Define forecast months =====================

earlyYM = makeYMList( ...
    2019, 1, ...
    2021, 12);

lateYM = makeYMList( ...
    2022, 1, ...
    2024, 12);

masterYM = [earlyYM; lateYM];

nForecastMonth = numel(masterYM);

if nForecastMonth ~= nYear * horizon

    error( ...
        'Forecast-month count does not equal nYear x horizon.');

end

%% ===================== Extract 2019-2024 observations =====================

[tfObs, obsIdx] = ismember(masterYM, ymRaw);

if any(~tfObs)

    missingYM = masterYM(~tfObs);

    error( ...
        'Raw dGWS file is missing target months: %s', ...
        mat2str(missingYM(:)'));

end

if numel(unique(obsIdx)) ~= numel(obsIdx)

    error( ...
        'Duplicate month matching detected in raw ym.');

end

% [pixel x 72 months]
obsValid72 = dGWS_valid(:, obsIdx);

if ~isequal( ...
        size(obsValid72), ...
        [nValid, nYear * horizon])

    error( ...
        'Unexpected observation matrix dimensions.');

end

%% ===================== Define pixel-year samples =====================

fprintf('\nConstructing pixel-year sample index...\n');

nSamplePerYear = nValid;
nSample = nValid * nYear;

% Sample order:
%
% 2019:
%   pixel 1
%   pixel 2
%   ...
%   pixel nValid
%
% 2020:
%   pixel 1
%   ...
%
% etc.

casePixelID = repmat( ...
    uint32((1:nValid)'), ...
    nYear, 1);

caseForecastYear = zeros( ...
    nSample, 1, 'uint16');

caseForecastOriginYM = zeros( ...
    nSample, 1, 'uint32');

for y = 1:nYear

    thisYear = forecastYears(y);

    rows = ...
        (y - 1) * nSamplePerYear + ...
        (1:nSamplePerYear);

    caseForecastYear(rows) = ...
        uint16(thisYear);

    caseForecastOriginYM(rows) = ...
        uint32((thisYear - 1) * 100 + 12);

end

caseLat = validLat(double(casePixelID));
caseLon = validLon(double(casePixelID));

%% ===================== Initialize Master =====================

Master = struct();

Master.pixel_id = casePixelID;

Master.lat = caseLat;
Master.lon = caseLon;

Master.forecast_year = caseForecastYear;
Master.forecast_origin_ym = caseForecastOriginYM;

Master.model_names = modelNames;

% Each row = one pixel-year
% Columns = 12 forecast months
Master.observation = NaN( ...
    nSample, horizon);

% Dimensions:
%   sample x forecast_month x model
Master.prediction = NaN( ...
    nSample, horizon, nModel);

% Historical scale quantities
Master.preorigin_q10 = NaN(nSample, 1);
Master.preorigin_q90 = NaN(nSample, 1);
Master.preorigin_iqr80 = NaN(nSample, 1);

% Forecast-performance metrics
Master.rmse = NaN(nSample, nModel);
Master.nrmse_q10q90 = NaN(nSample, nModel);

% Spatial lookup
Master.validLinearIndex = validLinearIndex;
Master.validLat = validLat;
Master.validLon = validLon;

Master.lat_grid = latGrid;
Master.lon_grid = lonGrid;

%% ===================== Build annual observation blocks =====================

fprintf('\nBuilding annual observation matrices...\n');

for y = 1:nYear

    rows = ...
        (y - 1) * nValid + ...
        (1:nValid);

    cols = ...
        (y - 1) * horizon + ...
        (1:horizon);

    % [pixel x 12]
    Master.observation(rows, :) = ...
        obsValid72(:, cols);

end

%% ===================== Calculate pre-origin Q10/Q90 =====================

fprintf('\nCalculating pre-origin Q10-Q90 scales...\n');

for y = 1:nYear

    thisYear = forecastYears(y);

    originYM = ...
        (thisYear - 1) * 100 + 12;

    rows = ...
        (y - 1) * nValid + ...
        (1:nValid);

    % IMPORTANT:
    % Only data available at or before the forecast origin.
    historyIdx = ...
        ymRaw <= originYM;

    if ~any(historyIdx)

        error( ...
            'No pre-origin history available for forecast year %d.', ...
            thisYear);

    end

    historyData = ...
        dGWS_valid(:, historyIdx);

    fprintf( ...
        '  Forecast %d: history %d-%d, %d months\n', ...
        thisYear, ...
        ymRaw(find(historyIdx, 1, 'first')), ...
        ymRaw(find(historyIdx, 1, 'last')), ...
        sum(historyIdx));

    [q10, q90] = ...
        rowQ10Q90(historyData);

    scale80 = q90 - q10;

    % Non-positive range is not a meaningful normalization scale.
    scale80( ...
        ~isfinite(scale80) | scale80 <= 0) = NaN;

    Master.preorigin_q10(rows) = q10;
    Master.preorigin_q90(rows) = q90;
    Master.preorigin_iqr80(rows) = scale80;

end

%% ===================== Load 9-model predictions =====================

fprintf('\nLoading and combining 9-model predictions...\n');

for m = 1:nModel

    modelName = sourceFiles(m).model;
    predVar = sourceFiles(m).predVar;

    fprintf( ...
        '\n[%d/%d] %s\n', ...
        m, nModel, modelName);

    %% ----- 2019-2021 -----

    fprintf('  Loading 2019-2021...\n');

    predEarly = loadPredictionAsValidGrid( ...
        sourceFiles(m).earlyFile, ...
        predVar, ...
        earlyYM, ...
        validLinearIndex, ...
        CommonValid, ...
        nLat, ...
        nLon);

    if ~isequal(size(predEarly), [nValid, 36])

        error( ...
            ['%s 2019-2021 valid-grid prediction ', ...
             'dimensions are unexpected.'], ...
            modelName);

    end

    %% ----- 2022-2024 -----

    fprintf('  Loading 2022-2024...\n');

    predLate = loadPredictionAsValidGrid( ...
        sourceFiles(m).lateFile, ...
        predVar, ...
        lateYM, ...
        validLinearIndex, ...
        CommonValid, ...
        nLat, ...
        nLon);

    if ~isequal(size(predLate), [nValid, 36])

        error( ...
            ['%s 2022-2024 valid-grid prediction ', ...
             'dimensions are unexpected.'], ...
            modelName);

    end

    %% ----- Combine to 2019-2024 -----

    % [pixel x 72 months]
    predValid72 = ...
        [predEarly, predLate];

    %% ----- Convert to pixel-year structure -----

    for y = 1:nYear

        rows = ...
            (y - 1) * nValid + ...
            (1:nValid);

        cols = ...
            (y - 1) * horizon + ...
            (1:horizon);

        % [pixel x 12]
        predBlock = ...
            predValid72(:, cols);

        Master.prediction(rows, :, m) = ...
            predBlock;

    end

    fprintf( ...
        '  Non-finite predictions: %d / %d\n', ...
        sum(~isfinite(predValid72(:))), ...
        numel(predValid72));

clear predEarly predLate predValid72 predBlock;

end

%% ===================== Calculate annual RMSE =====================

fprintf('\nCalculating annual 12-month RMSE...\n');

for m = 1:nModel

    predM = ...
        Master.prediction(:, :, m);

    obsM = ...
        Master.observation;

    % A valid annual metric requires all 12 observations
    % and all 12 predictions to be finite.
    completeYear = ...
        all(isfinite(obsM), 2) & ...
        all(isfinite(predM), 2);

    err = ...
        predM - obsM;

    rmseM = NaN(nSample, 1);

    rmseM(completeYear) = ...
        sqrt(mean( ...
        err(completeYear, :).^2, ...
        2));

    Master.rmse(:, m) = ...
        rmseM;

end

%% ===================== Calculate Q10-Q90 normalized RMSE =====================

fprintf('\nCalculating Q10-Q90 normalized RMSE...\n');

scale80 = ...
    Master.preorigin_iqr80;

validScale = ...
    isfinite(scale80) & ...
    scale80 > 0;

for m = 1:nModel

    metric = NaN(nSample, 1);

    validCase = ...
        validScale & ...
        isfinite(Master.rmse(:, m));

    metric(validCase) = ...
        Master.rmse(validCase, m) ./ ...
        scale80(validCase);

    Master.nrmse_q10q90(:, m) = ...
        metric;

end

%% ===================== Consistency checks =====================

fprintf('\nRunning consistency checks...\n');

if size(Master.observation, 1) ~= nSample || ...
        size(Master.observation, 2) ~= horizon

    error( ...
        'Master.observation dimensions are inconsistent.');

end

if ~isequal( ...
        size(Master.prediction), ...
        [nSample, horizon, nModel])

    error( ...
        'Master.prediction dimensions are inconsistent.');

end

if ~isequal( ...
        size(Master.rmse), ...
        [nSample, nModel])

    error( ...
        'Master.rmse dimensions are inconsistent.');

end

if ~isequal( ...
        size(Master.nrmse_q10q90), ...
        [nSample, nModel])

    error( ...
        'Master.nrmse_q10q90 dimensions are inconsistent.');

end

%% ----- Check number of samples per year -----

for y = 1:nYear

    thisYear = forecastYears(y);

    yearMask = ...
        double(Master.forecast_year) == thisYear;

    nThisYear = ...
        sum(yearMask);

    if nThisYear ~= nValid

        error( ...
            ['Forecast year %d has %d samples; ', ...
             'expected %d.'], ...
            thisYear, ...
            nThisYear, ...
            nValid);

    end

end

%% ----- Check origin month -----

expectedOriginYM = uint32( ...
    (double(Master.forecast_year) - 1) * 100 + 12);

if any( ...
        Master.forecast_origin_ym ~= ...
        expectedOriginYM)

    error( ...
        'forecast_origin_ym is inconsistent with forecast_year.');

end

%% ----- Verify Q90 >= Q10 -----

badQuantileOrder = ...
    isfinite(Master.preorigin_q10) & ...
    isfinite(Master.preorigin_q90) & ...
    Master.preorigin_q90 < Master.preorigin_q10;

if any(badQuantileOrder)

    error( ...
        'Detected pre-origin Q90 < Q10.');

end

fprintf('All consistency checks passed.\n');

%% ===================== Metadata =====================

Meta = struct();

Meta.script = ...
    's1_master_database_pixel_year.m';

Meta.database = ...
    ['2019-2024 pixel-year groundwater ', ...
     'forecast-performance database'];

Meta.rowDefinition = ...
    'One sample = one valid pixel x one forecast year.';

Meta.caseOrder = ...
    'forecast_year -> pixel_id';

Meta.forecastYears = ...
    forecastYears;

Meta.horizon = ...
    horizon;

Meta.nValid = ...
    nValid;

Meta.nYear = ...
    nYear;

Meta.nSample = ...
    nSample;

Meta.nModel = ...
    nModel;

Meta.modelNames = ...
    modelNames;

Meta.observationSource = ...
    rawDataFile;

Meta.observationVariable = ...
    'dGWS_valid';

Meta.predictionSourceNote = [ ...
    '2019-2021 predictions come from Manuscript 2 pseudo-forecast ', ...
    'experiments; 2022-2024 predictions come from the original ', ...
    'Manuscript 1 forecasting experiments.'];

Meta.performanceMetric = [ ...
    'Annual RMSE is calculated across the full 12-month forecast ', ...
    'year. NRMSE_Q10Q90 = annual RMSE / ', ...
    '(Q90(pre-origin dGWS) - Q10(pre-origin dGWS)).'];

Meta.normalizationNote = [ ...
    'For forecast year Y, Q10 and Q90 are calculated exclusively ', ...
    'from dGWS observations available through December of Y-1. ', ...
    'No target-year observation is used to define the scale.'];

Meta.missingDataRule = [ ...
    'Annual RMSE requires all 12 observations and all 12 model ', ...
    'predictions to be finite. Otherwise the annual RMSE and ', ...
    'normalized RMSE are NaN. Pre-origin quantiles use finite ', ...
    'historical dGWS values.'];

Meta.targetInterpretation = [ ...
    'Lower NRMSE_Q10Q90 indicates higher forecastability; ', ...
    'higher NRMSE_Q10Q90 indicates lower forecastability.'];

Meta.validationNote = [ ...
    'Subsequent meta-model training should use temporal holdout ', ...
    'and grouped cross-validation by pixel ID.'];

Meta.rawDataPeriod = ...
    sprintf('%d to %d', ...
    ymRaw(1), ...
    ymRaw(end));

Meta.forecastTargetPeriod = ...
    sprintf('%d to %d', ...
    masterYM(1), ...
    masterYM(end));

Meta.outputFile = ...
    outFile;

Master.Meta = Meta;
Master.source_files = sourceFiles;

%% ===================== Summary =====================

fprintf('\n');
fprintf('============================================================\n');
fprintf('PIXEL-YEAR MASTER DATABASE SUMMARY\n');
fprintf('============================================================\n');

fprintf( ...
    'Valid pixels             : %d\n', ...
    nValid);

fprintf( ...
    'Forecast years           : %s\n', ...
    mat2str(forecastYears));

fprintf( ...
    'Forecast horizon         : %d months\n', ...
    horizon);

fprintf( ...
    'Total pixel-year samples : %d\n', ...
    nSample);

fprintf( ...
    'Models                   : %d\n', ...
    nModel);

fprintf('\n');

fprintf( ...
    'Non-finite observations  : %d / %d\n', ...
    sum(~isfinite(Master.observation(:))), ...
    numel(Master.observation));

fprintf( ...
    'Invalid Q10-Q90 scales   : %d / %d\n', ...
    sum(~isfinite(Master.preorigin_iqr80)), ...
    nSample);

fprintf('\n');

for m = 1:nModel

    nBadPred = ...
        sum(~isfinite( ...
        Master.prediction(:, :, m)), ...
        'all');

    nBadRMSE = ...
        sum(~isfinite( ...
        Master.rmse(:, m)));

    nBadNRMSE = ...
        sum(~isfinite( ...
        Master.nrmse_q10q90(:, m)));

    fprintf( ...
        ['%-7s | bad monthly pred: %7d | ', ...
         'bad annual RMSE: %6d | ', ...
         'bad annual NRMSE: %6d\n'], ...
        modelNames{m}, ...
        nBadPred, ...
        nBadRMSE, ...
        nBadNRMSE);

end

fprintf('\n');

%% ----- Mean metric by model -----

fprintf('Global mean annual Q10-Q90 normalized RMSE:\n');

for m = 1:nModel

    x = ...
        Master.nrmse_q10q90(:, m);

    x = ...
        x(isfinite(x));

    fprintf( ...
        '  %-7s : %.4f\n', ...
        modelNames{m}, ...
        mean(x));

end

fprintf('============================================================\n');

%% ===================== Save =====================

fprintf('\nSaving database to:\n%s\n', outFile);

save( ...
    outFile, ...
    'Master', ...
    '-v7.3');

fprintf('Database saved successfully.\n');

%% ===================== Local functions =====================

function assertFileExists(filePath, labelText)

    if ~exist(filePath, 'file')

        error( ...
            '%s not found:\n%s', ...
            labelText, ...
            filePath);

    end

end


function predValid = loadPredictionAsValidGrid( ...
    filePath, ...
    predVar, ...
    expectedPredYM, ...
    rawValidLinearIndex, ...
    rawCommonValid, ...
    nLat, ...
    nLon)

    % Load into a local structure so variables from different
    % result files cannot overwrite each other.

    S = load( ...
        filePath, ...
        predVar, ...
        'predYM', ...
        'validLinearIndex', ...
        'CommonValid');

    if ~isfield(S, predVar)

        error( ...
            'File is missing prediction variable %s:\n%s', ...
            predVar, ...
            filePath);

    end

    if ~isfield(S, 'predYM')

        error( ...
            'File is missing predYM:\n%s', ...
            filePath);

    end

    if ~isfield(S, 'validLinearIndex')

        error( ...
            'File is missing validLinearIndex:\n%s', ...
            filePath);

    end

    if ~isfield(S, 'CommonValid')

        error( ...
            'File is missing CommonValid:\n%s', ...
            filePath);

    end

    predGrid = ...
        double(S.(predVar));

    filePredYM = ...
        double(S.predYM(:));

    fileValidLinearIndex = ...
        double(S.validLinearIndex(:));

    fileCommonValid = ...
        logical(S.CommonValid);

    expectedPredYM = ...
        double(expectedPredYM(:));

    rawValidLinearIndex = ...
        double(rawValidLinearIndex(:));

    %% ----- Time-index check -----

    if ~isequal( ...
            filePredYM, ...
            expectedPredYM)

        error( ...
            ['predYM mismatch in file:\n%s\n', ...
             'Expected:\n%s\n', ...
             'Found:\n%s'], ...
            filePath, ...
            mat2str(expectedPredYM(:)'), ...
            mat2str(filePredYM(:)'));

    end

    %% ----- Spatial-index check -----

    if ~isequal( ...
            fileValidLinearIndex, ...
            rawValidLinearIndex)

        error( ...
            'validLinearIndex mismatch in file:\n%s', ...
            filePath);

    end

    if ~isequal( ...
            fileCommonValid, ...
            rawCommonValid)

        error( ...
            'CommonValid mismatch in file:\n%s', ...
            filePath);

    end

    %% ----- Grid-dimension check -----

    if size(predGrid, 1) ~= nLat || ...
            size(predGrid, 2) ~= nLon

        error( ...
            ['Spatial dimensions of %s are [%d, %d], ', ...
             'expected [%d, %d]:\n%s'], ...
            predVar, ...
            size(predGrid, 1), ...
            size(predGrid, 2), ...
            nLat, ...
            nLon, ...
            filePath);

    end

    nPredMonth = ...
        numel(expectedPredYM);

    if size(predGrid, 3) ~= nPredMonth

        error( ...
            ['Time dimension of %s is %d, ', ...
             'expected %d:\n%s'], ...
            predVar, ...
            size(predGrid, 3), ...
            nPredMonth, ...
            filePath);

    end

    %% ----- Global grid -> valid pixels -----

    predFlat = ...
        reshape( ...
        predGrid, ...
        nLat * nLon, ...
        nPredMonth);

    predValid = ...
        predFlat( ...
        rawValidLinearIndex, :);

end


function [q10, q90] = rowQ10Q90(X)

    % Calculate row-wise Q10 and Q90.
    %
    % Fast path:
    %   if all values are finite, MATLAB prctile is applied directly.
    %
    % Missing-data path:
    %   finite values are extracted pixel by pixel.
    %
    % A minimum of 12 finite months is required.

    X = double(X);

    nRow = size(X, 1);

    q10 = NaN(nRow, 1);
    q90 = NaN(nRow, 1);

    if all(isfinite(X(:)))

        q10 = prctile(X, 10, 2);
        q90 = prctile(X, 90, 2);

        return

    end

    for i = 1:nRow

        x = X(i, :);

        x = ...
            x(isfinite(x));

        if numel(x) < 12
            continue
        end

        q10(i) = ...
            prctile(x, 10);

        q90(i) = ...
            prctile(x, 90);

    end

end


function ym = makeYMList( ...
    y1, m1, ...
    y2, m2)

    d1 = ...
        datetime(y1, m1, 1);

    d2 = ...
        datetime(y2, m2, 1);

    d = ...
        d1:calmonths(1):d2;

    ym = ...
        year(d(:)) * 100 + ...
        month(d(:));

    ym = ...
        double(ym(:));

end