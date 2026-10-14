%% Step 1: Build 2019-2024 master forecast-performance database
% File name:
%   s1_master_database.m
%
% Purpose:
%   Combine the 2019-2021 pseudo-forecast results and the original
%   2022-2024 forecast results from all 9 groundwater forecasting models
%   into one unified master database.
%
% Basic sampling unit:
%   pixel x forecast_year x forecast_lead
%
% Case order in the saved database:
%   forecast_year -> pixel_id -> lead
%
% Forecast years:
%   2019, 2020, 2021, 2022, 2023, 2024
%
% Models:
%   AR, ARIMA, SARIMA, SVM, RF, LSTM, SVMX, RFX, LSTMX
%
% Important design decisions:
%   1. This script does NOT construct hydrological-state features.
%   2. This script does NOT train the competence model.
%   3. This script does NOT define a final normalized competence metric.
%   4. It preserves the raw observation, prediction, signed error, and
%      absolute error for all 9 models.
%   5. It does NOT reduce the 9-model error profile to a winner label.
%   6. Each .mat model-result file is loaded into a local structure inside
%      a helper function, so same-named variables from 2019-2021 and
%      2022-2024 files cannot overwrite one another.
%
% Output:
%   E:\P1_SKLGB\M_manuscript2\s3_analysis\s1_master_database.mat
%
% Main saved variable:
%   Master
%
% Master contains:
%   pixel_id                 [nCase, 1] uint32
%   lat                      [nCase, 1] double
%   lon                      [nCase, 1] double
%   forecast_year            [nCase, 1] uint16
%   forecast_origin_year     [nCase, 1] uint16
%   forecast_origin_ym       [nCase, 1] uint32
%   lead                     [nCase, 1] uint8
%   target_ym                [nCase, 1] uint32
%   observation              [nCase, 1] double
%   prediction               [nCase, 9] double
%   signed_error             [nCase, 9] double
%   absolute_error           [nCase, 9] double
%   model_names              1 x 9 cell
%   validLinearIndex         [nValid, 1]
%   validLat                 [nValid, 1]
%   validLon                 [nValid, 1]
%   lat_grid                 global latitude vector
%   lon_grid                 global longitude vector
%   source_files             source-file metadata
%   Meta                     database metadata
%
% Error convention:
%   signed_error = prediction - observation
%   absolute_error = abs(prediction - observation)
%
% The final competence target (e.g., absolute error, normalized absolute
% error, etc.) should be defined later from these preserved quantities.

clc;
clearvars;

%% ===================== User settings =====================

analysisFolder = 'E:\P1_SKLGB\M_manuscript2\s3_analysis\';

if ~exist(analysisFolder, 'dir')
    mkdir(analysisFolder);
end

outFile = fullfile(analysisFolder, 's1_master_database.mat');

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

%% ===================== Define all source files =====================

% Each model has:
%   earlyFile : 2019-2021 result
%   lateFile  : 2022-2024 result
%   predVar   : prediction variable inside both files
%
% Files are deliberately handled one at a time inside helper functions.
% This avoids same-name variable overwriting between paired files.

sourceFiles = repmat(struct( ...
    'model', '', ...
    'predVar', '', ...
    'earlyFile', '', ...
    'lateFile', ''), nModel, 1);

% ----- Linear group -----

sourceFiles(1).model = 'AR';
sourceFiles(1).predVar = 'AR_Pred';
sourceFiles(1).earlyFile = fullfile( ...
    p2LinearFolder, 'AR_univariate_global_2019_2021.mat');
sourceFiles(1).lateFile = fullfile( ...
    p1LinearFolder, 'AR_univariate_global_200401_202412.mat');

sourceFiles(2).model = 'ARIMA';
sourceFiles(2).predVar = 'ARIMA_Pred';
sourceFiles(2).earlyFile = fullfile( ...
    p2LinearFolder, 'ARIMA_univariate_global_2019_2021.mat');
sourceFiles(2).lateFile = fullfile( ...
    p1LinearFolder, 'ARIMA_univariate_global_200401_202412_d0.mat');

sourceFiles(3).model = 'SARIMA';
sourceFiles(3).predVar = 'SARIMA_Pred';
sourceFiles(3).earlyFile = fullfile( ...
    p2LinearFolder, 'SARIMA_univariate_global_2019_2021.mat');
sourceFiles(3).lateFile = fullfile( ...
    p1LinearFolder, 'SARIMA_univariate_global_200401_202412_d0.mat');

% ----- Nonlinear groundwater-history-only group -----

sourceFiles(4).model = 'SVM';
sourceFiles(4).predVar = 'SVM_Pred';
sourceFiles(4).earlyFile = fullfile( ...
    p2NonlinearFolder, 'SVM_univariate_global_2019_2021.mat');
sourceFiles(4).lateFile = fullfile( ...
    p1NonlinearFolder, 'SVM_univariate_global_200401_202412.mat');

sourceFiles(5).model = 'RF';
sourceFiles(5).predVar = 'RF_Pred';
sourceFiles(5).earlyFile = fullfile( ...
    p2NonlinearFolder, 'RF_univariate_global_2019_2021.mat');
sourceFiles(5).lateFile = fullfile( ...
    p1NonlinearFolder, 'RF_univariate_global_200401_202412.mat');

sourceFiles(6).model = 'LSTM';
sourceFiles(6).predVar = 'LSTM_Pred';
sourceFiles(6).earlyFile = fullfile( ...
    p2NonlinearFolder, 'LSTM_univariate_global_2019_2021.mat');
sourceFiles(6).lateFile = fullfile( ...
    p1NonlinearFolder, 'LSTM_univariate_global_200401_202412.mat');

% ----- Nonlinear + exogenous-variable group -----

sourceFiles(7).model = 'SVMX';
sourceFiles(7).predVar = 'SVMX_Pred';
sourceFiles(7).earlyFile = fullfile( ...
    p2XFolder, 'SVMX_multivariate_global_2019_2021.mat');
sourceFiles(7).lateFile = fullfile( ...
    p1XFolder, 'SVMX_multivariate_global_200401_202412.mat');

sourceFiles(8).model = 'RFX';
sourceFiles(8).predVar = 'RFX_Pred';
sourceFiles(8).earlyFile = fullfile( ...
    p2XFolder, 'RFX_multivariate_global_2019_2021.mat');
sourceFiles(8).lateFile = fullfile( ...
    p1XFolder, 'RFX_multivariate_global_200401_202412.mat');

sourceFiles(9).model = 'LSTMX';
sourceFiles(9).predVar = 'LSTMX_Pred';
sourceFiles(9).earlyFile = fullfile( ...
    p2XFolder, 'LSTMX_multivariate_global_2019_2021.mat');
sourceFiles(9).lateFile = fullfile( ...
    p1XFolder, 'LSTMX_multivariate_global_200401_202412.mat');

%% ===================== Check that all source files exist =====================

fprintf('Checking source files...\n');

assertFileExists(rawDataFile, 'Raw dGWS data file');

for m = 1:nModel
    assertFileExists(sourceFiles(m).earlyFile, ...
        sprintf('%s 2019-2021 result', sourceFiles(m).model));

    assertFileExists(sourceFiles(m).lateFile, ...
        sprintf('%s 2022-2024 result', sourceFiles(m).model));
end

fprintf('All required source files were found.\n');

%% ===================== Load raw observation/reference data =====================

fprintf('\nLoading raw dGWS/reference data...\n');

D = load(rawDataFile, ...
    'dGWS_valid', ...
    'CommonMask', 'CommonValid', ...
    'validLinearIndex', 'validLat', 'validLon', ...
    'lat', 'lon', 'ym');

requiredRawFields = { ...
    'dGWS_valid', 'CommonMask', 'CommonValid', ...
    'validLinearIndex', 'validLat', 'validLon', ...
    'lat', 'lon', 'ym'};

for k = 1:numel(requiredRawFields)
    if ~isfield(D, requiredRawFields{k})
        error('Raw data file is missing required variable: %s', ...
            requiredRawFields{k});
    end
end

dGWS_valid = double(D.dGWS_valid);
CommonMask = D.CommonMask;
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

if numel(validLinearIndex) ~= nValid
    error(['Raw data inconsistency: size(dGWS_valid,1) = %d, but ', ...
        'numel(validLinearIndex) = %d.'], ...
        nValid, numel(validLinearIndex));
end

if numel(validLat) ~= nValid || numel(validLon) ~= nValid
    error('Raw data inconsistency: validLat/validLon length does not equal nValid.');
end

if numel(ymRaw) ~= nRawMonth
    error(['Raw data inconsistency: size(dGWS_valid,2) = %d, but ', ...
        'numel(ym) = %d.'], ...
        nRawMonth, numel(ymRaw));
end

if ~isequal(size(CommonValid), [nLat, nLon])
    error('CommonValid dimensions do not match lat/lon grid dimensions.');
end

fprintf('Valid grid cells : %d\n', nValid);
fprintf('Raw months       : %d\n', nRawMonth);
fprintf('Grid size        : %d x %d\n', nLat, nLon);

%% ===================== Define expected forecast months =====================

earlyYM = makeYMList(2019, 1, 2021, 12);
lateYM = makeYMList(2022, 1, 2024, 12);
masterYM = [earlyYM; lateYM];

nYear = numel(forecastYears);
nMasterMonth = numel(masterYM);

if nMasterMonth ~= nYear * horizon
    error('masterYM length is inconsistent with forecastYears x horizon.');
end

% Locate true dGWS observations for 2019-2024 in the raw input file.
[tfObs, obsIdx] = ismember(masterYM, ymRaw);

if any(~tfObs)
    missingYM = masterYM(~tfObs);
    error('Raw dGWS file is missing required target months: %s', ...
        mat2str(missingYM(:)'));
end

if numel(unique(obsIdx)) ~= numel(obsIdx)
    error('Duplicate target-month matching detected in raw ym.');
end

obsValid72 = dGWS_valid(:, obsIdx);

if ~isequal(size(obsValid72), [nValid, nMasterMonth])
    error('Unexpected observation matrix dimensions after month extraction.');
end

%% ===================== Build case-index arrays =====================

fprintf('\nBuilding case-index arrays...\n');

nCasePerYear = nValid * horizon;
nCase = nCasePerYear * nYear;

% Within each forecast year:
%   pixel 1: lead 1...12
%   pixel 2: lead 1...12
%   ...
% Then move to the next forecast year.
basePixelID = repelem(uint32((1:nValid)'), horizon);
baseLead = repmat(uint8((1:horizon)'), nValid, 1);

casePixelID = repmat(basePixelID, nYear, 1);
caseLead = repmat(baseLead, nYear, 1);

caseForecastYear = zeros(nCase, 1, 'uint16');
caseForecastOriginYear = zeros(nCase, 1, 'uint16');
caseForecastOriginYM = zeros(nCase, 1, 'uint32');
caseTargetYM = zeros(nCase, 1, 'uint32');

for y = 1:nYear
    yearVal = forecastYears(y);

    row1 = (y - 1) * nCasePerYear + 1;
    row2 = y * nCasePerYear;
    rows = row1:row2;

    monthCols = (y - 1) * horizon + (1:horizon);
    yearYM = uint32(masterYM(monthCols));

    caseForecastYear(rows) = uint16(yearVal);
    caseForecastOriginYear(rows) = uint16(yearVal - 1);
    caseForecastOriginYM(rows) = uint32((yearVal - 1) * 100 + 12);

    % Repeating a 12-month vector nValid times matches the
    % pixel-major / lead-minor ordering above.
    caseTargetYM(rows) = repmat(yearYM(:), nValid, 1);
end

caseLat = validLat(double(casePixelID));
caseLon = validLon(double(casePixelID));

% Convert the [pixel x 72 months] observation matrix to the same
% year -> pixel -> lead case ordering.
caseObservation = flattenYearPixelLead( ...
    obsValid72, nValid, nYear, horizon);

if numel(caseObservation) ~= nCase
    error('Observation case-vector length is inconsistent with nCase.');
end

%% ===================== Initialize master database =====================

Master = struct();

Master.pixel_id = casePixelID;
Master.lat = caseLat;
Master.lon = caseLon;

Master.forecast_year = caseForecastYear;
Master.forecast_origin_year = caseForecastOriginYear;
Master.forecast_origin_ym = caseForecastOriginYM;
Master.lead = caseLead;
Master.target_ym = caseTargetYM;

Master.observation = caseObservation;

Master.model_names = modelNames;
Master.prediction = NaN(nCase, nModel);

% Save compact per-pixel spatial lookup information as well.
Master.validLinearIndex = validLinearIndex;
Master.validLat = validLat;
Master.validLon = validLon;
Master.lat_grid = latGrid;
Master.lon_grid = lonGrid;

%% ===================== Load and combine predictions model by model =====================

fprintf('\nLoading and combining 9-model predictions...\n');

for m = 1:nModel
    modelName = sourceFiles(m).model;
    predVar = sourceFiles(m).predVar;

    fprintf('\n[%d/%d] %s\n', m, nModel, modelName);
    fprintf('  Loading 2019-2021...\n');

    predEarly = loadPredictionAsValidGrid( ...
        sourceFiles(m).earlyFile, ...
        predVar, ...
        earlyYM, ...
        validLinearIndex, ...
        CommonValid, ...
        nLat, nLon);

    fprintf('  Loading 2022-2024...\n');

    predLate = loadPredictionAsValidGrid( ...
        sourceFiles(m).lateFile, ...
        predVar, ...
        lateYM, ...
        validLinearIndex, ...
        CommonValid, ...
        nLat, nLon);

    if ~isequal(size(predEarly), [nValid, 36])
        error('%s 2019-2021 valid-grid prediction dimensions are unexpected.', ...
            modelName);
    end

    if ~isequal(size(predLate), [nValid, 36])
        error('%s 2022-2024 valid-grid prediction dimensions are unexpected.', ...
            modelName);
    end

    predValid72 = [predEarly, predLate];

    Master.prediction(:, m) = flattenYearPixelLead( ...
        predValid72, nValid, nYear, horizon);

    earlyNan = sum(~isfinite(predEarly(:)));
    lateNan = sum(~isfinite(predLate(:)));

    fprintf('  Non-finite predictions 2019-2021: %d / %d\n', ...
        earlyNan, numel(predEarly));
    fprintf('  Non-finite predictions 2022-2024: %d / %d\n', ...
        lateNan, numel(predLate));

    clear predEarly predLate predValid72
end

%% ===================== Compute basic realized errors =====================

fprintf('\nComputing realized errors...\n');

% MATLAB implicit expansion:
%   prediction [nCase x 9]
%   observation [nCase x 1]
Master.signed_error = Master.prediction - Master.observation;
Master.absolute_error = abs(Master.signed_error);

%% ===================== Internal consistency checks =====================

fprintf('\nRunning master-database consistency checks...\n');

if size(Master.prediction, 1) ~= nCase || size(Master.prediction, 2) ~= nModel
    error('Master.prediction dimensions are inconsistent.');
end

if ~isequal(size(Master.signed_error), [nCase, nModel])
    error('Master.signed_error dimensions are inconsistent.');
end

if ~isequal(size(Master.absolute_error), [nCase, nModel])
    error('Master.absolute_error dimensions are inconsistent.');
end

if numel(Master.observation) ~= nCase
    error('Master.observation length is inconsistent.');
end

% Check year counts.
for y = 1:nYear
    thisYear = forecastYears(y);
    nThisYear = sum(double(Master.forecast_year) == thisYear);

    if nThisYear ~= nCasePerYear
        error('Forecast year %d has %d cases; expected %d.', ...
            thisYear, nThisYear, nCasePerYear);
    end
end

% Check each year contains every lead exactly nValid times.
for y = 1:nYear
    thisYear = forecastYears(y);
    yearMask = double(Master.forecast_year) == thisYear;

    for h = 1:horizon
        nThisLead = sum(yearMask & double(Master.lead) == h);

        if nThisLead ~= nValid
            error(['Year %d, lead %d has %d cases; expected exactly ', ...
                '%d valid pixels.'], ...
                thisYear, h, nThisLead, nValid);
        end
    end
end

% Check target month is compatible with forecast year and lead.
expectedTargetYM = ...
    uint32(double(Master.forecast_year) * 100 + double(Master.lead));

if any(Master.target_ym ~= expectedTargetYM)
    error('target_ym is inconsistent with forecast_year/lead.');
end

% Check origin is December of the preceding year.
expectedOriginYM = ...
    uint32((double(Master.forecast_year) - 1) * 100 + 12);

if any(Master.forecast_origin_ym ~= expectedOriginYM)
    error('forecast_origin_ym is inconsistent with forecast_year.');
end

fprintf('All consistency checks passed.\n');

%% ===================== Metadata =====================

Meta = struct();

Meta.script = 's1_master_database.m';
Meta.database = '2019-2024 groundwater forecast-performance master database';
Meta.rowDefinition = 'One case = pixel x forecast_year x forecast_lead.';
Meta.caseOrder = 'forecast_year -> pixel_id -> lead.';
Meta.forecastYears = forecastYears;
Meta.horizon = horizon;
Meta.nValid = nValid;
Meta.nYear = nYear;
Meta.nModel = nModel;
Meta.nCase = nCase;
Meta.modelNames = modelNames;

Meta.observationSource = rawDataFile;
Meta.observationVariable = 'dGWS_valid';

Meta.predictionSourceNote = [ ...
    '2019-2021 results are pseudo-forecast experiments generated for Manuscript 2; ', ...
    '2022-2024 results are the original Manuscript 1 forecast outputs.'];

Meta.sameVariableNameSafety = [ ...
    'Each result file is loaded into a local structure inside ', ...
    'loadPredictionAsValidGrid; paired files are never loaded into the ', ...
    'base workspace simultaneously.'];

Meta.predictionShape = '[nCase, 9], columns follow model_names.';
Meta.errorConvention = 'signed_error = prediction - observation; absolute_error = abs(signed_error).';
Meta.finalCompetenceTarget = [ ...
    'Not fixed in Step 2. Raw prediction, observation, signed error, and ', ...
    'absolute error are preserved so normalized or alternative competence ', ...
    'labels can be constructed later without rerunning the 9 forecast models.'];

Meta.winnerNote = [ ...
    'No winner/best-model label is stored as the primary target in Step 2; ', ...
    'winner, ranking, minimum error, and regret should be derived later from ', ...
    'the full 9-model continuous error profile.'];

Meta.forecastOriginNote = [ ...
    'For each forecast year Y, forecast_origin_year = Y-1 and ', ...
    'forecast_origin_ym = December of Y-1. Lead resets to 1 at each new ', ...
    'forecast year; 2019-2021 and 2022-2024 are not treated as one ', ...
    'continuous 72-step forecast.'];

Meta.futureUse = [ ...
    'This master database is a data warehouse. Prospective competence-model ', ...
    'training must still enforce expanding temporal splits: ', ...
    '2019-2021 -> 2022, 2019-2022 -> 2023, 2019-2023 -> 2024.'];

Meta.rawDataPeriod = sprintf('%d to %d', ymRaw(1), ymRaw(end));
Meta.masterTargetPeriod = sprintf('%d to %d', masterYM(1), masterYM(end));
Meta.outputFile = outFile;

Master.Meta = Meta;
Master.source_files = sourceFiles;

%% ===================== Summary =====================

fprintf('\n===================== MASTER DATABASE SUMMARY =====================\n');
fprintf('Valid pixels            : %d\n', nValid);
fprintf('Forecast years          : %s\n', mat2str(forecastYears));
fprintf('Forecast leads/year     : %d\n', horizon);
fprintf('Total cases             : %d\n', nCase);
fprintf('Models                  : %d\n', nModel);
fprintf('Observation non-finite  : %d / %d\n', ...
    sum(~isfinite(Master.observation)), numel(Master.observation));

for m = 1:nModel
    nBadPred = sum(~isfinite(Master.prediction(:, m)));
    nBadAbsErr = sum(~isfinite(Master.absolute_error(:, m)));

    fprintf('%-7s non-finite prediction: %7d / %d | absolute error: %7d / %d\n', ...
        modelNames{m}, ...
        nBadPred, nCase, ...
        nBadAbsErr, nCase);
end

fprintf('===================================================================\n');

%% ===================== Save =====================

fprintf('\nSaving master database to:\n%s\n', outFile);

save(outFile, 'Master', '-v7.3');

fprintf('Master database saved successfully.\n');

%% ===================== Local functions =====================

function assertFileExists(filePath, labelText)
    if ~exist(filePath, 'file')
        error('%s not found:\n%s', labelText, filePath);
    end
end

function predValid = loadPredictionAsValidGrid( ...
    filePath, predVar, expectedPredYM, rawValidLinearIndex, ...
    rawCommonValid, nLat, nLon)

    % Load each result file into its own local structure.
    % This prevents same-named variables from different files from
    % overwriting one another.
    S = load(filePath, ...
        predVar, ...
        'predYM', ...
        'validLinearIndex', ...
        'CommonValid');

    if ~isfield(S, predVar)
        error('File is missing prediction variable %s:\n%s', ...
            predVar, filePath);
    end

    if ~isfield(S, 'predYM')
        error('File is missing predYM:\n%s', filePath);
    end

    if ~isfield(S, 'validLinearIndex')
        error('File is missing validLinearIndex:\n%s', filePath);
    end

    if ~isfield(S, 'CommonValid')
        error('File is missing CommonValid:\n%s', filePath);
    end

    predGrid = double(S.(predVar));
    filePredYM = double(S.predYM(:));
    fileValidLinearIndex = double(S.validLinearIndex(:));
    fileCommonValid = logical(S.CommonValid);

    expectedPredYM = double(expectedPredYM(:));
    rawValidLinearIndex = double(rawValidLinearIndex(:));

    if ~isequal(filePredYM, expectedPredYM)
        error(['predYM mismatch in file:\n%s\nExpected:\n%s\nFound:\n%s'], ...
            filePath, ...
            mat2str(expectedPredYM(:)'), ...
            mat2str(filePredYM(:)'));
    end

    if ~isequal(fileValidLinearIndex, rawValidLinearIndex)
        error('validLinearIndex mismatch in file:\n%s', filePath);
    end

    if ~isequal(fileCommonValid, rawCommonValid)
        error('CommonValid mismatch in file:\n%s', filePath);
    end

    if size(predGrid, 1) ~= nLat || size(predGrid, 2) ~= nLon
        error(['Spatial dimensions of %s in file are [%d, %d], ', ...
            'expected [%d, %d]:\n%s'], ...
            predVar, size(predGrid, 1), size(predGrid, 2), ...
            nLat, nLon, filePath);
    end

    nPredMonth = numel(expectedPredYM);

    if size(predGrid, 3) ~= nPredMonth
        error(['Time dimension of %s is %d, expected %d:\n%s'], ...
            predVar, size(predGrid, 3), nPredMonth, filePath);
    end

    % Convert [lat x lon x time] global fields to
    % [valid_pixel x time], preserving validLinearIndex order.
    predFlat = reshape(predGrid, nLat * nLon, nPredMonth);
    predValid = predFlat(rawValidLinearIndex, :);

    clear S predGrid predFlat
end

function v = flattenYearPixelLead(M, nValid, nYear, horizon)
    % Convert:
    %   M [nValid x (nYear*horizon)]
    %
    % to case order:
    %   year 1:
    %       pixel 1, lead 1...horizon
    %       pixel 2, lead 1...horizon
    %       ...
    %   year 2:
    %       ...
    %
    % Output:
    %   v [nValid*nYear*horizon x 1]

    M = double(M);

    if ~isequal(size(M), [nValid, nYear * horizon])
        error(['flattenYearPixelLead received matrix size [%d, %d], ', ...
            'expected [%d, %d].'], ...
            size(M, 1), size(M, 2), ...
            nValid, nYear * horizon);
    end

    nCasePerYear = nValid * horizon;
    v = NaN(nCasePerYear * nYear, 1);

    for y = 1:nYear
        cols = (y - 1) * horizon + (1:horizon);

        block = M(:, cols);              % [pixel x lead]
        blockVector = reshape(block.', [], 1);  % pixel-major, lead-minor

        row1 = (y - 1) * nCasePerYear + 1;
        row2 = y * nCasePerYear;

        v(row1:row2) = blockVector;
    end
end

function ym = makeYMList(y1, m1, y2, m2)
    d1 = datetime(y1, m1, 1);
    d2 = datetime(y2, m2, 1);
    d = d1:calmonths(1):d2;

    ym = year(d(:)) * 100 + month(d(:));
    ym = double(ym(:));
end
