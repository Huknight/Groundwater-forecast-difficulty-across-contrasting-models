%% s10_forecastability_robustness_checks.m
%
% ============================================================
% ROBUSTNESS CHECKS FOR GROUNDWATER FORECASTABILITY MANUSCRIPT
% ============================================================
%
% This script addresses three reviewer-style concerns without
% retraining M1/M2/M3 or rerunning SHAP:
%
%   A. Amplitude sensitivity
%      Does the strong cross-model forecast-difficulty commonality
%      persist after explicitly controlling the pre-origin dGWS
%      amplitude A = Q90 - Q10?
%
%      Two complementary checks are used:
%        A1. Rank-residualized partial Spearman correlations
%            controlling A.
%        A2. Within-amplitude-quartile Spearman correlations.
%
%   B. Low-amplitude denominator sensitivity
%      Does the commonality persist after removing the lowest
%      1%, 5%, and 10% of Q90-Q10 amplitudes?
%
%   C. Spatial-block bootstrap
%      Are the M3-vs-M1 and M3-vs-M2 improvements still robust
%      when spatial autocorrelation is addressed by resampling
%      latitude-longitude blocks rather than individual pixels?
%
%      Designs compared:
%        - original pixel-cluster bootstrap
%        - 2 x 2 degree blocks
%        - 5 x 5 degree blocks
%        - 10 x 10 degree blocks
%
% IMPORTANT
% ---------
% 1. The script uses the SAME target definitions as the manuscript:
%       NRMSE_Q10Q90
%       log1p(NRMSE_Q10Q90)
%
% 2. Spatial bootstrap uses ONLY the fully held-out years:
%       2022, 2023, 2024
%
% 3. The nine forecast-model targets are treated as the fixed set
%    of studied forecasting configurations, exactly as in the
%    original s6_M1_M2_M3_comparison.m analysis.
%
% 4. No RF fitting is performed here. Existing held-out predictions
%    are loaded from s5 MAT files when available. If those MAT files
%    are unavailable, the script falls back to prediction CSV folders
%    or the three supplied ZIP archives.
%
% OUTPUT FOLDER
% -------------
%   s10_forecastability_robustness_outputs\
%
% Main outputs:
%   01_AmplitudeControl.csv
%   02_AmplitudeStrata.csv
%   03_LowAmplitudeSensitivity.csv
%   04_LowAmplitudeSummary.csv
%   05_SpatialBlockBootstrap.csv
%   06_SpatialBlockStructure.csv
%   s10_forecastability_robustness.mat
%   logs\s10_robustness_*.txt
%
% ============================================================

clc;
clearvars;
close all;


%% ============================================================
% 1. PATHS
% ============================================================

analysisFolder = ...
    'E:\P1_SKLGB\M_manuscript2\s3_analysis\';

scriptFolder = ...
    fileparts(mfilename('fullpath'));

if isempty(scriptFolder)
    scriptFolder = pwd;
end

masterFile = ...
    firstExistingFile({ ...
    fullfile(analysisFolder,'s1_master_database_pixel_year.mat'), ...
    fullfile(scriptFolder,'s1_master_database_pixel_year.mat')});

m1File = ...
    firstExistingFileOptional({ ...
    fullfile(analysisFolder,'s5_RF_meta_model.mat'), ...
    fullfile(scriptFolder,'s5_RF_meta_model.mat')});

m2File = ...
    firstExistingFileOptional({ ...
    fullfile(analysisFolder,'s5_RF_meta_model_M2_hydro.mat'), ...
    fullfile(scriptFolder,'s5_RF_meta_model_M2_hydro.mat')});

m3File = ...
    firstExistingFileOptional({ ...
    fullfile(analysisFolder,'s5_RF_meta_model_M3_dgws_hydro.mat'), ...
    fullfile(scriptFolder,'s5_RF_meta_model_M3_dgws_hydro.mat')});

outputFolder = ...
    fullfile( ...
    analysisFolder, ...
    's10_forecastability_robustness_outputs');

if ~exist(analysisFolder,'dir')
    % Portable fallback: if the original Windows analysis path is not
    % available, write outputs beside this script.
    outputFolder = ...
        fullfile( ...
        scriptFolder, ...
        's10_forecastability_robustness_outputs');
end

if ~exist(outputFolder,'dir')
    mkdir(outputFolder);
end

logFolder = fullfile(outputFolder,'logs');
if ~exist(logFolder,'dir')
    mkdir(logFolder);
end

unzipFolder = fullfile(outputFolder,'_unzipped_predictions');


%% ============================================================
% 2. SETTINGS
% ============================================================

scriptVersion = ...
    '2026-10-09_robustness_v1';

expectedNPixel = 17284;
expectedNModel = 9;

expectedModelNames = { ...
    'AR', ...
    'ARIMA', ...
    'SARIMA', ...
    'SVM', ...
    'RF', ...
    'LSTM', ...
    'SVMX', ...
    'RFX', ...
    'LSTMX'};

allYears = 2019:2024;
outerTestYears = [2022 2023 2024];

% Amplitude strata for sensitivity analysis.
amplitudeQuartileEdges = [0 25 50 75 100];

% Remove the lowest x% amplitude cases.
lowAmplitudeRemovalPct = [0 1 5 10];

% Spatial bootstrap settings.
blockSizesDeg = [2 5 10];
nBootstrap = 2000;
bootstrapSeed = 20261009;
confidenceLevel = 0.95;
alpha = 1 - confidenceLevel;

comparisonNames = {'M3_vs_M1','M3_vs_M2'};
referenceVariant = [1 2];
candidateVariant = [3 3];
variantNames = {'M1_dGWS','M2_Hydro','M3_Combined'};

% Numerical audit tolerances. These are NOT hypothesis thresholds;
% they only detect accidental data/path misalignment.
expectedPooledNRMSESpearmanApprox = 0.882;
correlationAuditTolerance = 0.010;

expectedDeltaM3vsM1 = -0.00847;
expectedDeltaM3vsM2 = -0.00664;
deltaAuditTolerance = 0.0010;


%% ============================================================
% 3. LOG
% ============================================================

runTimeStamp = datestr(now,'yyyymmdd_HHMMSS');
logFile = fullfile( ...
    logFolder, ...
    ['s10_robustness_',runTimeStamp,'.txt']);

diary(logFile);
diary on;

fprintf('\n');
fprintf('============================================================\n');
fprintf('s10_forecastability_robustness_checks.m\n');
fprintf('Version : %s\n',scriptVersion);
fprintf('Started : %s\n',datestr(now));
fprintf('============================================================\n');


%% ============================================================
% 4. LOAD MASTER DATABASE
% ============================================================

fprintf('\nLoading master database:\n%s\n',masterFile);

S = load(masterFile,'Master');
if ~isfield(S,'Master')
    error('Master structure not found in %s.',masterFile);
end
Master = S.Master;
clear S;

requiredMasterFields = { ...
    'pixel_id', ...
    'forecast_year', ...
    'lat', ...
    'lon', ...
    'model_names', ...
    'rmse', ...
    'nrmse_q10q90', ...
    'preorigin_q10', ...
    'preorigin_q90'};

for i = 1:numel(requiredMasterFields)
    if ~isfield(Master,requiredMasterFields{i})
        error('Master is missing required field: %s', ...
            requiredMasterFields{i});
    end
end

pixelID = double(Master.pixel_id(:));
forecastYear = double(Master.forecast_year(:));
lat = double(Master.lat(:));
lon = double(Master.lon(:));

nSample = numel(pixelID);

modelNames = cellstr(string(Master.model_names(:)))';
if numel(modelNames) ~= expectedNModel
    error('Expected %d model names; found %d.', ...
        expectedNModel,numel(modelNames));
end
if ~isequal(modelNames,expectedModelNames)
    error('Unexpected model order. Found: %s', ...
        strjoin(modelNames,', '));
end

RMSE = forceSampleByModel( ...
    double(Master.rmse), ...
    nSample, ...
    expectedNModel, ...
    'Master.rmse');

NRMSE = forceSampleByModel( ...
    double(Master.nrmse_q10q90), ...
    nSample, ...
    expectedNModel, ...
    'Master.nrmse_q10q90');

preQ10 = double(Master.preorigin_q10(:));
preQ90 = double(Master.preorigin_q90(:));
amplitude = preQ90 - preQ10;

if isfield(Master,'preorigin_iqr80')
    ampStored = double(Master.preorigin_iqr80(:));
    maxAmpDifference = max(abs(amplitude - ampStored));
    fprintf('Max |(Q90-Q10)-stored IQR80| = %.3g\n',maxAmpDifference);
else
    maxAmpDifference = NaN;
end

if any(~isfinite(RMSE(:))) || any(RMSE(:) < 0)
    error('RMSE contains invalid values.');
end
if any(~isfinite(NRMSE(:))) || any(NRMSE(:) < 0)
    error('NRMSE contains invalid values.');
end
if any(~isfinite(amplitude)) || any(amplitude <= 0)
    error('Pre-origin Q90-Q10 amplitude contains invalid/non-positive values.');
end

% Exact target-definition audit: NRMSE must equal RMSE/(Q90-Q10).
nrmseReconstructed = RMSE ./ amplitude;
maxNRMSEDifference = max(abs(nrmseReconstructed(:)-NRMSE(:)));
if maxNRMSEDifference > 1e-10
    error(['Target-definition audit failed: max |NRMSE - RMSE/(Q90-Q10)| ', ...
        '= %.3g.'],maxNRMSEDifference);
end
clear nrmseReconstructed;
fprintf('Max |NRMSE - RMSE/(Q90-Q10)| = %.3g\n',maxNRMSEDifference);

actualYears = unique(forecastYear)';
if ~isequal(actualYears,allYears)
    error('Expected years %s; found %s.', ...
        mat2str(allYears),mat2str(actualYears));
end

uniquePixels = unique(pixelID);
nPixel = numel(uniquePixels);
if nPixel ~= expectedNPixel
    error('Expected %d unique pixels; found %d.', ...
        expectedNPixel,nPixel);
end

fprintf('\nMaster database verified:\n');
fprintf('  Samples       : %d\n',nSample);
fprintf('  Pixels        : %d\n',nPixel);
fprintf('  Years         : %s\n',mat2str(actualYears));
fprintf('  Models        : %s\n',strjoin(modelNames,', '));
fprintf('  Amplitude min : %.6g\n',min(amplitude));
fprintf('  Amplitude P5  : %.6g\n',prctile(amplitude,5));
fprintf('  Amplitude P50 : %.6g\n',median(amplitude));


%% ============================================================
% 5. PIXEL-YEAR ROW MAP
% ============================================================

nAllYear = numel(allYears);
PixelYearRowMapAll = NaN(nPixel,nAllYear);

for iy = 1:nAllYear
    thisYear = allYears(iy);
    rows = find(forecastYear == thisYear);

    if numel(rows) ~= nPixel
        error('Year %d contains %d rows; expected %d.', ...
            thisYear,numel(rows),nPixel);
    end

    [sortedPixel,ord] = sort(pixelID(rows));
    sortedRows = rows(ord);

    if ~isequal(sortedPixel,uniquePixels)
        error('Pixel population/order mismatch in %d.',thisYear);
    end

    PixelYearRowMapAll(:,iy) = sortedRows;
end

PixelYearRowMapAll = double(PixelYearRowMapAll);

primaryYearLoc = NaN(1,numel(outerTestYears));
for iy = 1:numel(outerTestYears)
    primaryYearLoc(iy) = find(allYears == outerTestYears(iy),1);
end
PixelYearRowMapTest = PixelYearRowMapAll(:,primaryYearLoc);

firstTestRows = PixelYearRowMapTest(:,1);
pixelLat = lat(firstTestRows);
pixelLon = lon(firstTestRows);

% Verify coordinates do not change across years.
for iy = 2:numel(outerTestYears)
    rows = PixelYearRowMapTest(:,iy);
    if any(abs(lat(rows)-pixelLat) > 1e-10) || ...
            any(abs(lon(rows)-pixelLon) > 1e-10)
        error('Pixel coordinates differ among outer-test years.');
    end
end


%% ============================================================
% 6. BASELINE COMMONALITY AUDIT
% ============================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf('BASELINE COMMONALITY AUDIT\n');
fprintf('============================================================\n');

allMask = ismember(forecastYear,allYears);
primaryMask = ismember(forecastYear,outerTestYears);

baseNRMSEAll = correlationSummary(NRMSE(allMask,:),'Spearman');
baseRMSEAll = correlationSummary(RMSE(allMask,:),'Spearman');
baseNRMSEPrimary = correlationSummary(NRMSE(primaryMask,:),'Spearman');
baseRMSEPrimary = correlationSummary(RMSE(primaryMask,:),'Spearman');

fprintf('Pooled 2019-2024 NRMSE mean pairwise Spearman = %.6f\n', ...
    baseNRMSEAll.Mean);
fprintf('Pooled 2019-2024 RMSE  mean pairwise Spearman = %.6f\n', ...
    baseRMSEAll.Mean);
fprintf('Pooled 2022-2024 NRMSE mean pairwise Spearman = %.6f\n', ...
    baseNRMSEPrimary.Mean);
fprintf('Pooled 2022-2024 RMSE  mean pairwise Spearman = %.6f\n', ...
    baseRMSEPrimary.Mean);

if abs(baseNRMSEAll.Mean - expectedPooledNRMSESpearmanApprox) > ...
        correlationAuditTolerance
    error(['Baseline pooled NRMSE correlation does not reproduce the ', ...
        'manuscript result closely enough. Expected approximately %.3f; ', ...
        'found %.6f. Check the input database.'], ...
        expectedPooledNRMSESpearmanApprox,baseNRMSEAll.Mean);
end


%% ============================================================
% 7. A — AMPLITUDE-CONTROL SENSITIVITY
% ============================================================
%
% Rank residualization:
%   1. Convert each target and amplitude to ranks.
%   2. Regress each target-rank on amplitude-rank.
%   3. Correlate residual ranks across the nine target models.
%
% This is a transparent partial-Spearman-style control for A = Q90-Q10.
% ============================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf('A. AMPLITUDE-CONTROL SENSITIVITY\n');
fprintf('============================================================\n');

scopeNames = { ...
    'Pooled_2019_2024', ...
    'Pooled_2022_2024', ...
    'Year_2019', ...
    'Year_2020', ...
    'Year_2021', ...
    'Year_2022', ...
    'Year_2023', ...
    'Year_2024'};

scopeMasks = cell(size(scopeNames));
scopeMasks{1} = ismember(forecastYear,allYears);
scopeMasks{2} = ismember(forecastYear,outerTestYears);
for iy = 1:numel(allYears)
    scopeMasks{2+iy} = forecastYear == allYears(iy);
end

nScope = numel(scopeNames);

A_Scope = cell(nScope,1);
A_N = NaN(nScope,1);
A_AmpMedian = NaN(nScope,1);
A_NRMSE_Unadjusted_Mean = NaN(nScope,1);
A_NRMSE_Adjusted_Mean = NaN(nScope,1);
A_NRMSE_Adjusted_Median = NaN(nScope,1);
A_NRMSE_Adjusted_Min = NaN(nScope,1);
A_NRMSE_Adjusted_Max = NaN(nScope,1);
A_RMSE_Unadjusted_Mean = NaN(nScope,1);
A_RMSE_Adjusted_Mean = NaN(nScope,1);
A_RMSE_Adjusted_Median = NaN(nScope,1);
A_RMSE_Adjusted_Min = NaN(nScope,1);
A_RMSE_Adjusted_Max = NaN(nScope,1);

for is = 1:nScope
    mask = scopeMasks{is};

    A_Scope{is} = scopeNames{is};
    A_N(is) = sum(mask);
    A_AmpMedian(is) = median(amplitude(mask));

    s = correlationSummary(NRMSE(mask,:),'Spearman');
    A_NRMSE_Unadjusted_Mean(is) = s.Mean;

    Cadj = rankResidualCorrelation(NRMSE(mask,:),amplitude(mask));
    s = offDiagonalSummary(Cadj);
    A_NRMSE_Adjusted_Mean(is) = s.Mean;
    A_NRMSE_Adjusted_Median(is) = s.Median;
    A_NRMSE_Adjusted_Min(is) = s.Minimum;
    A_NRMSE_Adjusted_Max(is) = s.Maximum;

    s = correlationSummary(RMSE(mask,:),'Spearman');
    A_RMSE_Unadjusted_Mean(is) = s.Mean;

    Cadj = rankResidualCorrelation(RMSE(mask,:),amplitude(mask));
    s = offDiagonalSummary(Cadj);
    A_RMSE_Adjusted_Mean(is) = s.Mean;
    A_RMSE_Adjusted_Median(is) = s.Median;
    A_RMSE_Adjusted_Min(is) = s.Minimum;
    A_RMSE_Adjusted_Max(is) = s.Maximum;
end

AmplitudeControlTable = table( ...
    A_Scope, ...
    A_N, ...
    A_AmpMedian, ...
    A_NRMSE_Unadjusted_Mean, ...
    A_NRMSE_Adjusted_Mean, ...
    A_NRMSE_Adjusted_Median, ...
    A_NRMSE_Adjusted_Min, ...
    A_NRMSE_Adjusted_Max, ...
    A_RMSE_Unadjusted_Mean, ...
    A_RMSE_Adjusted_Mean, ...
    A_RMSE_Adjusted_Median, ...
    A_RMSE_Adjusted_Min, ...
    A_RMSE_Adjusted_Max, ...
    'VariableNames', { ...
    'Scope', ...
    'N', ...
    'AmplitudeMedian', ...
    'NRMSE_Spearman_Unadjusted_Mean', ...
    'NRMSE_PartialSpearman_AmpAdjusted_Mean', ...
    'NRMSE_PartialSpearman_AmpAdjusted_Median', ...
    'NRMSE_PartialSpearman_AmpAdjusted_Min', ...
    'NRMSE_PartialSpearman_AmpAdjusted_Max', ...
    'RMSE_Spearman_Unadjusted_Mean', ...
    'RMSE_PartialSpearman_AmpAdjusted_Mean', ...
    'RMSE_PartialSpearman_AmpAdjusted_Median', ...
    'RMSE_PartialSpearman_AmpAdjusted_Min', ...
    'RMSE_PartialSpearman_AmpAdjusted_Max'});

writetable(AmplitudeControlTable, ...
    fullfile(outputFolder,'01_AmplitudeControl.csv'));

fprintf('\nAmplitude-control summary:\n');
disp(AmplitudeControlTable(1:2,:));


%% ============================================================
% 8. A2 — WITHIN-AMPLITUDE-QUARTILE CORRELATIONS
% ============================================================

fprintf('\nCalculating within-amplitude-quartile correlations...\n');

StrataRows = {};
rowCounter = 0;

for is = 1:nScope
    maskScope = scopeMasks{is};
    ampScope = amplitude(maskScope);
    nrmseScope = NRMSE(maskScope,:);
    rmseScope = RMSE(maskScope,:);

    edges = prctile(ampScope,amplitudeQuartileEdges);

    for ib = 1:(numel(amplitudeQuartileEdges)-1)
        if ib == 1
            keep = ampScope >= edges(ib) & ampScope <= edges(ib+1);
        else
            keep = ampScope > edges(ib) & ampScope <= edges(ib+1);
        end

        sN = correlationSummary(nrmseScope(keep,:),'Spearman');
        sR = correlationSummary(rmseScope(keep,:),'Spearman');

        rowCounter = rowCounter + 1;
        StrataRows(rowCounter,:) = { ...
            scopeNames{is}, ...
            ib, ...
            amplitudeQuartileEdges(ib), ...
            amplitudeQuartileEdges(ib+1), ...
            edges(ib), ...
            edges(ib+1), ...
            sum(keep), ...
            median(ampScope(keep)), ...
            sN.Mean, ...
            sN.Median, ...
            sN.Minimum, ...
            sN.Maximum, ...
            sR.Mean, ...
            sR.Median, ...
            sR.Minimum, ...
            sR.Maximum};
    end
end

AmplitudeStrataTable = cell2table( ...
    StrataRows, ...
    'VariableNames', { ...
    'Scope', ...
    'AmplitudeBin', ...
    'PercentileLow', ...
    'PercentileHigh', ...
    'AmplitudeLow', ...
    'AmplitudeHigh', ...
    'N', ...
    'AmplitudeMedian', ...
    'NRMSE_Spearman_Mean', ...
    'NRMSE_Spearman_Median', ...
    'NRMSE_Spearman_Min', ...
    'NRMSE_Spearman_Max', ...
    'RMSE_Spearman_Mean', ...
    'RMSE_Spearman_Median', ...
    'RMSE_Spearman_Min', ...
    'RMSE_Spearman_Max'});

AmplitudeStrataTable = convertNumericTableColumns( ...
    AmplitudeStrataTable, ...
    2:16);

writetable(AmplitudeStrataTable, ...
    fullfile(outputFolder,'02_AmplitudeStrata.csv'));


%% ============================================================
% 9. B — LOW-AMPLITUDE DENOMINATOR SENSITIVITY
% ============================================================
%
% Thresholds are computed WITHIN each analysis scope. This avoids
% imposing a threshold from one forecast year on another year whose
% pre-origin reference period is slightly longer.
% ============================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf('B. LOW-AMPLITUDE DENOMINATOR SENSITIVITY\n');
fprintf('============================================================\n');

LowRows = {};
rowCounter = 0;

for is = 1:nScope
    maskScope = scopeMasks{is};
    ampScope = amplitude(maskScope);
    nrmseScope = NRMSE(maskScope,:);
    rmseScope = RMSE(maskScope,:);

    for ir = 1:numel(lowAmplitudeRemovalPct)
        pctRemove = lowAmplitudeRemovalPct(ir);

        if pctRemove == 0
            threshold = -Inf;
            keep = true(size(ampScope));
        else
            threshold = prctile(ampScope,pctRemove);
            keep = ampScope > threshold;
        end

        sN = correlationSummary(nrmseScope(keep,:),'Spearman');
        sR = correlationSummary(rmseScope(keep,:),'Spearman');

        rowCounter = rowCounter + 1;
        LowRows(rowCounter,:) = { ...
            scopeNames{is}, ...
            pctRemove, ...
            threshold, ...
            numel(ampScope), ...
            sum(keep), ...
            100*sum(keep)/numel(ampScope), ...
            min(ampScope(keep)), ...
            median(ampScope(keep)), ...
            sN.Mean, ...
            sN.Median, ...
            sN.Minimum, ...
            sN.Maximum, ...
            sR.Mean, ...
            sR.Median, ...
            sR.Minimum, ...
            sR.Maximum};
    end
end

LowAmplitudeTable = cell2table( ...
    LowRows, ...
    'VariableNames', { ...
    'Scope', ...
    'RemovedLowestAmplitudePct', ...
    'AmplitudeThreshold', ...
    'NBefore', ...
    'NAfter', ...
    'PercentRetained', ...
    'AmplitudeMinRetained', ...
    'AmplitudeMedianRetained', ...
    'NRMSE_Spearman_Mean', ...
    'NRMSE_Spearman_Median', ...
    'NRMSE_Spearman_Min', ...
    'NRMSE_Spearman_Max', ...
    'RMSE_Spearman_Mean', ...
    'RMSE_Spearman_Median', ...
    'RMSE_Spearman_Min', ...
    'RMSE_Spearman_Max'});

LowAmplitudeTable = convertNumericTableColumns( ...
    LowAmplitudeTable, ...
    2:16);

writetable(LowAmplitudeTable, ...
    fullfile(outputFolder,'03_LowAmplitudeSensitivity.csv'));

% Compact manuscript-oriented summary: pooled all years and primary years.
summaryScope = {'Pooled_2019_2024','Pooled_2022_2024'};
summaryMask = ismember(LowAmplitudeTable.Scope,summaryScope);
LowAmplitudeSummary = LowAmplitudeTable(summaryMask,:);

writetable(LowAmplitudeSummary, ...
    fullfile(outputFolder,'04_LowAmplitudeSummary.csv'));

fprintf('\nLow-amplitude summary:\n');
disp(LowAmplitudeSummary(:,{ ...
    'Scope', ...
    'RemovedLowestAmplitudePct', ...
    'NAfter', ...
    'NRMSE_Spearman_Mean'}));


%% ============================================================
% 10. LOAD M1/M2/M3 HELD-OUT PREDICTIONS
% ============================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf('LOADING HELD-OUT M1 / M2 / M3 PREDICTIONS\n');
fprintf('============================================================\n');

[PredictionLog, predictionSourceDescription] = ...
    loadPredictionLogAllVariants( ...
    m1File, ...
    m2File, ...
    m3File, ...
    analysisFolder, ...
    scriptFolder, ...
    unzipFolder, ...
    pixelID, ...
    forecastYear, ...
    lat, ...
    lon, ...
    NRMSE, ...
    modelNames, ...
    outerTestYears);

fprintf('Prediction source: %s\n',predictionSourceDescription);

Ylog = log1p(NRMSE);

testMaskAll = ismember(forecastYear,outerTestYears);
for v = 1:numel(variantNames)
    if any(~isfinite(PredictionLog{v}(testMaskAll,:)),'all')
        error('%s contains non-finite held-out predictions.', ...
            variantNames{v});
    end
end


%% ============================================================
% 11. RECONSTRUCT PIXEL x YEAR x TARGET HELD-OUT CUBES
% ============================================================

nTestYear = numel(outerTestYears);

ObservedLogCube = NaN(nPixel,nTestYear,expectedNModel);
PredictionLogCube = cell(numel(variantNames),1);
for v = 1:numel(variantNames)
    PredictionLogCube{v} = NaN(nPixel,nTestYear,expectedNModel);
end

for iy = 1:nTestYear
    rows = PixelYearRowMapTest(:,iy);
    ObservedLogCube(:,iy,:) = reshape( ...
        Ylog(rows,:), ...
        [nPixel 1 expectedNModel]);

    for v = 1:numel(variantNames)
        PredictionLogCube{v}(:,iy,:) = reshape( ...
            PredictionLog{v}(rows,:), ...
            [nPixel 1 expectedNModel]);
    end
end

SquaredErrorCube = cell(numel(variantNames),1);
for v = 1:numel(variantNames)
    SquaredErrorCube{v} = ...
        (PredictionLogCube{v} - ObservedLogCube).^2;
end


%% ============================================================
% 12. ORIGINAL POINT-ESTIMATE AUDIT
% ============================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf('M1/M2/M3 POINT-ESTIMATE AUDIT\n');
fprintf('============================================================\n');

observedOverall = observedOverallComparisons( ...
    SquaredErrorCube, ...
    comparisonNames, ...
    referenceVariant, ...
    candidateVariant);

disp(observedOverall);

rowM31 = strcmp(observedOverall.Comparison,'M3_vs_M1');
rowM32 = strcmp(observedOverall.Comparison,'M3_vs_M2');

if abs(observedOverall.ObservedMeanDeltaRMSE_Log(rowM31) - ...
        expectedDeltaM3vsM1) > deltaAuditTolerance
    error(['M3-vs-M1 point estimate was not reproduced. ', ...
        'Expected approximately %.5f; found %.5f.'], ...
        expectedDeltaM3vsM1, ...
        observedOverall.ObservedMeanDeltaRMSE_Log(rowM31));
end

if abs(observedOverall.ObservedMeanDeltaRMSE_Log(rowM32) - ...
        expectedDeltaM3vsM2) > deltaAuditTolerance
    error(['M3-vs-M2 point estimate was not reproduced. ', ...
        'Expected approximately %.5f; found %.5f.'], ...
        expectedDeltaM3vsM2, ...
        observedOverall.ObservedMeanDeltaRMSE_Log(rowM32));
end

fprintf('Original M3 comparison point estimates reproduced.\n');


%% ============================================================
% 13. C — SPATIAL-BLOCK BOOTSTRAP
% ============================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf('C. SPATIAL-BLOCK BOOTSTRAP\n');
fprintf('B = %d\n',nBootstrap);
fprintf('============================================================\n');

BootstrapTables = {};
StructureRows = {};
bootTableCounter = 0;
structureCounter = 0;

% ----- Original pixel-cluster bootstrap -----
pixelGroup = (1:nPixel)';

fprintf('\nRunning original pixel-cluster bootstrap...\n');

[Tboot,Sblock] = spatialClusterBootstrap( ...
    SquaredErrorCube, ...
    pixelGroup, ...
    'PixelCluster', ...
    0, ...
    comparisonNames, ...
    referenceVariant, ...
    candidateVariant, ...
    variantNames, ...
    nBootstrap, ...
    bootstrapSeed, ...
    alpha);

bootTableCounter = bootTableCounter + 1;
BootstrapTables{bootTableCounter} = Tboot;

structureCounter = structureCounter + 1;
StructureRows(structureCounter,:) = structToStructureRow(Sblock);

% ----- Spatial latitude-longitude blocks -----
for ib = 1:numel(blockSizesDeg)
    blockSize = blockSizesDeg(ib);

    [spatialGroup,nBlock] = makeSpatialBlockGroups( ...
        pixelLat,pixelLon,blockSize);

    fprintf('\nRunning %.0f-degree spatial-block bootstrap (%d blocks)...\n', ...
        blockSize,nBlock);

    [Tboot,Sblock] = spatialClusterBootstrap( ...
        SquaredErrorCube, ...
        spatialGroup, ...
        sprintf('SpatialBlock_%gdeg',blockSize), ...
        blockSize, ...
        comparisonNames, ...
        referenceVariant, ...
        candidateVariant, ...
        variantNames, ...
        nBootstrap, ...
        bootstrapSeed + round(blockSize*1000), ...
        alpha);

    bootTableCounter = bootTableCounter + 1;
    BootstrapTables{bootTableCounter} = Tboot;

    structureCounter = structureCounter + 1;
    StructureRows(structureCounter,:) = structToStructureRow(Sblock);
end

SpatialBootstrapTable = vertcat(BootstrapTables{:});

SpatialBlockStructure = cell2table( ...
    StructureRows, ...
    'VariableNames', { ...
    'Design', ...
    'BlockSizeDeg', ...
    'NClusters', ...
    'MeanPixelsPerCluster', ...
    'MedianPixelsPerCluster', ...
    'MinPixelsPerCluster', ...
    'MaxPixelsPerCluster'});

SpatialBlockStructure = convertNumericTableColumns( ...
    SpatialBlockStructure, ...
    2:7);

writetable(SpatialBootstrapTable, ...
    fullfile(outputFolder,'05_SpatialBlockBootstrap.csv'));

writetable(SpatialBlockStructure, ...
    fullfile(outputFolder,'06_SpatialBlockStructure.csv'));

fprintf('\nSpatial bootstrap results:\n');
disp(SpatialBootstrapTable(:,{ ...
    'Design', ...
    'Comparison', ...
    'ObservedMeanDeltaRMSE_Log', ...
    'ObservedMeanRMSEImprovementPct_Log', ...
    'DeltaRMSE_CI_Lower', ...
    'DeltaRMSE_CI_Upper', ...
    'ProbabilityCandidateBetter_RMSE', ...
    'BootstrapP_TwoSided_RMSE'}));


%% ============================================================
% 14. WRITE TEXT SUMMARY
% ============================================================

summaryFile = fullfile(outputFolder,'s10_Robustness_Summary.txt');
fid = fopen(summaryFile,'w');
if fid < 0
    error('Cannot create summary file: %s',summaryFile);
end
fprintf(fid,'============================================================\n');
fprintf(fid,'S10 FORECASTABILITY ROBUSTNESS SUMMARY\n');
fprintf(fid,'Version: %s\n',scriptVersion);
fprintf(fid,'Created: %s\n',datestr(now));
fprintf(fid,'============================================================\n\n');

fprintf(fid,'BASELINE COMMONALITY\n');
fprintf(fid,'Pooled 2019-2024 NRMSE mean pairwise Spearman: %.6f\n', ...
    baseNRMSEAll.Mean);
fprintf(fid,'Pooled 2019-2024 RMSE  mean pairwise Spearman: %.6f\n', ...
    baseRMSEAll.Mean);
fprintf(fid,'Pooled 2022-2024 NRMSE mean pairwise Spearman: %.6f\n', ...
    baseNRMSEPrimary.Mean);
fprintf(fid,'Pooled 2022-2024 RMSE  mean pairwise Spearman: %.6f\n\n', ...
    baseRMSEPrimary.Mean);

fprintf(fid,'AMPLITUDE-ADJUSTED COMMONALITY\n');
for i = 1:2
    fprintf(fid,'%s\n',AmplitudeControlTable.Scope{i});
    fprintf(fid,'  NRMSE unadjusted mean Spearman : %.6f\n', ...
        AmplitudeControlTable.NRMSE_Spearman_Unadjusted_Mean(i));
    fprintf(fid,'  NRMSE amplitude-adjusted mean  : %.6f\n', ...
        AmplitudeControlTable.NRMSE_PartialSpearman_AmpAdjusted_Mean(i));
    fprintf(fid,'  RMSE unadjusted mean Spearman  : %.6f\n', ...
        AmplitudeControlTable.RMSE_Spearman_Unadjusted_Mean(i));
    fprintf(fid,'  RMSE amplitude-adjusted mean   : %.6f\n', ...
        AmplitudeControlTable.RMSE_PartialSpearman_AmpAdjusted_Mean(i));
end
fprintf(fid,'\n');

fprintf(fid,'LOW-AMPLITUDE REMOVAL\n');
for i = 1:height(LowAmplitudeSummary)
    fprintf(fid,'%s | remove lowest %g%% | N=%d | mean NRMSE Spearman=%.6f\n', ...
        LowAmplitudeSummary.Scope{i}, ...
        LowAmplitudeSummary.RemovedLowestAmplitudePct(i), ...
        LowAmplitudeSummary.NAfter(i), ...
        LowAmplitudeSummary.NRMSE_Spearman_Mean(i));
end
fprintf(fid,'\n');

fprintf(fid,'SPATIAL-BLOCK BOOTSTRAP\n');
for i = 1:height(SpatialBootstrapTable)
    fprintf(fid,['%s | %s | DeltaRMSE=%.8f | improvement=%.4f%% | ', ...
        '95%% CI [%.8f, %.8f] | P(candidate better)=%.4f | p=%.6g\n'], ...
        SpatialBootstrapTable.Design{i}, ...
        SpatialBootstrapTable.Comparison{i}, ...
        SpatialBootstrapTable.ObservedMeanDeltaRMSE_Log(i), ...
        SpatialBootstrapTable.ObservedMeanRMSEImprovementPct_Log(i), ...
        SpatialBootstrapTable.DeltaRMSE_CI_Lower(i), ...
        SpatialBootstrapTable.DeltaRMSE_CI_Upper(i), ...
        SpatialBootstrapTable.ProbabilityCandidateBetter_RMSE(i), ...
        SpatialBootstrapTable.BootstrapP_TwoSided_RMSE(i));
end

fclose(fid);


%% ============================================================
% 15. SAVE MAT
% ============================================================

Robustness = struct();
Robustness.Meta.script_version = scriptVersion;
Robustness.Meta.created = datestr(now);
Robustness.Meta.master_file = masterFile;
Robustness.Meta.prediction_source = predictionSourceDescription;
Robustness.Meta.all_years = allYears;
Robustness.Meta.outer_test_years = outerTestYears;
Robustness.Meta.model_names = modelNames;
Robustness.Meta.block_sizes_deg = blockSizesDeg;
Robustness.Meta.bootstrap_replicates = nBootstrap;
Robustness.Meta.bootstrap_seed = bootstrapSeed;
Robustness.Meta.confidence_level = confidenceLevel;
Robustness.Meta.amplitude_definition = 'preorigin_q90 - preorigin_q10';
Robustness.Meta.rank_residualization = [ ...
    'Ranks of each error target are linearly residualized against ', ...
    'rank(Q90-Q10), then residual Pearson correlations are reported.'];

Robustness.Baseline.NRMSE_AllYears = baseNRMSEAll;
Robustness.Baseline.RMSE_AllYears = baseRMSEAll;
Robustness.Baseline.NRMSE_PrimaryYears = baseNRMSEPrimary;
Robustness.Baseline.RMSE_PrimaryYears = baseRMSEPrimary;
Robustness.AmplitudeControl = AmplitudeControlTable;
Robustness.AmplitudeStrata = AmplitudeStrataTable;
Robustness.LowAmplitude = LowAmplitudeTable;
Robustness.LowAmplitudeSummary = LowAmplitudeSummary;
Robustness.ObservedComparison = observedOverall;
Robustness.SpatialBootstrap = SpatialBootstrapTable;
Robustness.SpatialBlockStructure = SpatialBlockStructure;

save( ...
    fullfile(outputFolder,'s10_forecastability_robustness.mat'), ...
    'Robustness', ...
    '-v7.3');


%% ============================================================
% 16. FINAL MESSAGE
% ============================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf('S10 ROBUSTNESS CHECKS COMPLETED\n');
fprintf('Finished : %s\n',datestr(now));
fprintf('Output   : %s\n',outputFolder);
fprintf('Summary  : %s\n',summaryFile);
fprintf('Log      : %s\n',logFile);
fprintf('============================================================\n');

diary off;


%% ============================================================
% LOCAL FUNCTIONS
% ============================================================

function filePath = firstExistingFile(candidates)
    filePath = '';
    for i = 1:numel(candidates)
        if exist(candidates{i},'file')
            filePath = candidates{i};
            return
        end
    end
    error('Required file not found. Checked:\n%s',strjoin(candidates,'\n'));
end


function filePath = firstExistingFileOptional(candidates)
    filePath = '';
    for i = 1:numel(candidates)
        if exist(candidates{i},'file')
            filePath = candidates{i};
            return
        end
    end
end


function X = forceSampleByModel(X,nSample,nModel,labelText)
    if isequal(size(X),[nSample nModel])
        return
    elseif isequal(size(X),[nModel nSample])
        X = X';
    else
        error('%s dimensions are %s; expected [%d %d].', ...
            labelText,mat2str(size(X)),nSample,nModel);
    end
end


function S = correlationSummary(X,typeName)
    C = corr(X,'Type',typeName,'Rows','pairwise');
    S = offDiagonalSummary(C);
    S.Matrix = C;
end


function S = offDiagonalSummary(C)
    n = size(C,1);
    mask = triu(true(n),1);
    v = C(mask);
    v = v(isfinite(v));

    S = struct();
    S.Mean = mean(v);
    S.Median = median(v);
    S.Minimum = min(v);
    S.Maximum = max(v);
    S.Std = std(v);
    S.NPair = numel(v);
end


function C = rankResidualCorrelation(X,z)
    % Partial-Spearman-style correlation controlling one covariate.
    % All variables are transformed to ranks. Each target rank is
    % residualized against the centered amplitude rank, then residuals
    % are correlated with ordinary Pearson correlation.

    if size(X,1) ~= numel(z)
        error('rankResidualCorrelation: sample count mismatch.');
    end

    z = z(:);
    valid = isfinite(z) & all(isfinite(X),2);
    X = X(valid,:);
    z = z(valid);

    rz = tiedrank(z);
    rz = rz - mean(rz);

    R = NaN(size(X));

    denom = sum(rz.^2);
    if denom <= 0
        error('Amplitude rank has zero variance.');
    end

    for j = 1:size(X,2)
        ry = tiedrank(X(:,j));
        ry = ry - mean(ry);
        beta = sum(rz .* ry) / denom;
        R(:,j) = ry - beta .* rz;
    end

    C = corr(R,'Type','Pearson','Rows','pairwise');
end


function [PredictionLog,sourceDescription] = loadPredictionLogAllVariants( ...
    m1File,m2File,m3File,analysisFolder,scriptFolder,unzipFolder, ...
    pixelID,forecastYear,lat,lon,NRMSE,modelNames,outerTestYears)

    nSample = numel(pixelID);
    nModel = numel(modelNames);

    PredictionLog = cell(3,1);

    % Preferred route: use the exact Results structures consumed by the
    % original s6 comparison script.
    if ~isempty(m1File) && ~isempty(m2File) && ~isempty(m3File)
        files = {m1File,m2File,m3File};
        labels = {'M1','M2','M3'};

        for v = 1:3
            A = load(files{v},'Results');
            if ~isfield(A,'Results') || ...
                    ~isfield(A.Results,'prediction_log')
                error('%s MAT file lacks Results.prediction_log.',labels{v});
            end

            P = double(A.Results.prediction_log);
            P = forceSampleByModel(P,nSample,nModel, ...
                [labels{v},'.Results.prediction_log']);

            if isfield(A.Results,'pixel_id') && ...
                    ~isequal(double(A.Results.pixel_id(:)),pixelID)
                error('%s result pixel_id does not align with Master.',labels{v});
            end
            if isfield(A.Results,'forecast_year') && ...
                    ~isequal(double(A.Results.forecast_year(:)),forecastYear)
                error('%s result forecast_year does not align with Master.',labels{v});
            end

            PredictionLog{v} = P;
        end

        sourceDescription = 's5 Results MAT files';
        return
    end

    fprintf(['s5 result MAT files were not all found. Falling back to ', ...
        'prediction CSV folders / ZIP archives.\n']);

    % Locate or extract CSV prediction folders.
    folderM1 = locatePredictionFolder( ...
        analysisFolder,scriptFolder,unzipFolder, ...
        'M1','predictions.zip');

    folderM2 = locatePredictionFolder( ...
        analysisFolder,scriptFolder,unzipFolder, ...
        'M2','predictionsM2.zip');

    folderM3 = locatePredictionFolder( ...
        analysisFolder,scriptFolder,unzipFolder, ...
        'M3','predictionsM3.zip');

    folders = {folderM1,folderM2,folderM3};
    prefixes = {'','M2_','M3_'};

    for v = 1:3
        P = NaN(nSample,nModel);

        for iy = 1:numel(outerTestYears)
            thisYear = outerTestYears(iy);
            rows = find(forecastYear == thisYear);
            [sortedPixel,ord] = sort(pixelID(rows));
            sortedRows = rows(ord);

            for m = 1:nModel
                if v == 1
                    filename = sprintf('Predictions_%d_%s.csv', ...
                        thisYear,modelNames{m});
                else
                    filename = sprintf('Predictions_%s%d_%s.csv', ...
                        prefixes{v},thisYear,modelNames{m});
                end

                filePath = fullfile(folders{v},filename);
                if ~exist(filePath,'file')
                    error('Prediction CSV not found: %s',filePath);
                end

                T = readtable(filePath);

                required = { ...
                    'PixelID','Lat','Lon','TestYear', ...
                    'Observed_NRMSE','Observed_LogTarget','Predicted_LogTarget'};
                for k = 1:numel(required)
                    if ~ismember(required{k},T.Properties.VariableNames)
                        error('%s missing column %s.',filename,required{k});
                    end
                end

                [csvPixel,csvOrd] = sort(double(T.PixelID(:)));
                if ~isequal(csvPixel,sortedPixel)
                    error('%s PixelID population does not align with Master.',filename);
                end

                if any(double(T.TestYear(csvOrd)) ~= thisYear)
                    error('%s contains unexpected TestYear values.',filename);
                end

                if max(abs(double(T.Lat(csvOrd))-lat(sortedRows))) > 1e-8 || ...
                        max(abs(double(T.Lon(csvOrd))-lon(sortedRows))) > 1e-8
                    error('%s coordinates do not align with Master.',filename);
                end

                observedLogMaster = log1p(NRMSE(sortedRows,m));
                if max(abs(double(T.Observed_LogTarget(csvOrd))- ...
                        observedLogMaster)) > 1e-8
                    error('%s observed log targets do not align with Master.',filename);
                end

                P(sortedRows,m) = double(T.Predicted_LogTarget(csvOrd));
            end
        end

        PredictionLog{v} = P;
    end

    sourceDescription = 'held-out prediction CSV files / supplied ZIP archives';
end


function predictionFolder = locatePredictionFolder( ...
    analysisFolder,scriptFolder,unzipFolder,variantLabel,zipName)

    switch variantLabel
        case 'M1'
            folderCandidates = { ...
                fullfile(analysisFolder,'s5_RF_meta_model_outputs','predictions'), ...
                fullfile(scriptFolder,'predictions')};
        case 'M2'
            folderCandidates = { ...
                fullfile(analysisFolder,'s5_RF_meta_model_M2_hydro_outputs','predictions')};
        case 'M3'
            folderCandidates = { ...
                fullfile(analysisFolder,'s5_RF_meta_model_M3_dgws_hydro_outputs','predictions')};
        otherwise
            error('Unknown variant label: %s',variantLabel);
    end

    for i = 1:numel(folderCandidates)
        if exist(folderCandidates{i},'dir')
            predictionFolder = folderCandidates{i};
            return
        end
    end

    zipCandidates = { ...
        fullfile(analysisFolder,zipName), ...
        fullfile(scriptFolder,zipName)};

    zipPath = '';
    for i = 1:numel(zipCandidates)
        if exist(zipCandidates{i},'file')
            zipPath = zipCandidates{i};
            break
        end
    end

    if isempty(zipPath)
        error(['Could not locate %s prediction folder or ZIP. ', ...
            'Expected ZIP name: %s'],variantLabel,zipName);
    end

    variantUnzip = fullfile(unzipFolder,variantLabel);
    expectedFolder = fullfile(variantUnzip,'predictions');

    if ~exist(expectedFolder,'dir')
        if ~exist(variantUnzip,'dir')
            mkdir(variantUnzip);
        end
        fprintf('Extracting %s to %s\n',zipPath,variantUnzip);
        unzip(zipPath,variantUnzip);
    end

    if ~exist(expectedFolder,'dir')
        error('ZIP extraction did not create expected folder: %s',expectedFolder);
    end

    predictionFolder = expectedFolder;
end


function T = observedOverallComparisons( ...
    squaredErrorCube,comparisonNames,referenceVariant,candidateVariant)

    nComparison = numel(comparisonNames);
    nModel = size(squaredErrorCube{1},3);

    Comparison = comparisonNames(:);
    ObservedMeanDeltaRMSE_Log = NaN(nComparison,1);
    ObservedMeanRMSEImprovementPct_Log = NaN(nComparison,1);

    for ic = 1:nComparison
        refV = referenceVariant(ic);
        candV = candidateVariant(ic);

        rmseRef = NaN(nModel,1);
        rmseCand = NaN(nModel,1);

        for m = 1:nModel
            tempRef = squaredErrorCube{refV}(:,:,m);
            tempCand = squaredErrorCube{candV}(:,:,m);
            rmseRef(m) = sqrt(mean(tempRef(:)));
            rmseCand(m) = sqrt(mean(tempCand(:)));
        end

        ObservedMeanDeltaRMSE_Log(ic) = mean(rmseCand-rmseRef);
        ObservedMeanRMSEImprovementPct_Log(ic) = ...
            mean(100*(rmseRef-rmseCand)./rmseRef);
    end

    T = table( ...
        Comparison, ...
        ObservedMeanDeltaRMSE_Log, ...
        ObservedMeanRMSEImprovementPct_Log);
end


function [groupIndex,nGroup] = makeSpatialBlockGroups(lat,lon,blockSize)
    % Fixed global block origin at (-90,-180).
    latBin = floor((lat(:)+90) ./ blockSize);
    lonBin = floor((lon(:)+180) ./ blockSize);

    nLonBin = ceil(360/blockSize);
    rawID = latBin .* nLonBin + lonBin;

    [~,~,groupIndex] = unique(rawID,'sorted');
    nGroup = max(groupIndex);
end


function [T,S] = spatialClusterBootstrap( ...
    squaredErrorCube,groupIndex,designName,blockSizeDeg, ...
    comparisonNames,referenceVariant,candidateVariant,variantNames, ...
    nBootstrap,seed,alpha)

    nVariant = numel(squaredErrorCube);
    nPixel = size(squaredErrorCube{1},1);
    nYear = size(squaredErrorCube{1},2);
    nModel = size(squaredErrorCube{1},3);

    groupIndex = groupIndex(:);
    if numel(groupIndex) ~= nPixel
        error('Cluster-index length does not equal pixel count.');
    end

    groupNames = unique(groupIndex);
    nGroup = numel(groupNames);

    % Recode to 1:nGroup if needed.
    [~,~,groupIndex] = unique(groupIndex,'sorted');

    pixelCountByGroup = accumarray(groupIndex,1,[nGroup 1],@sum,0);
    observationCountByGroup = pixelCountByGroup * nYear;

    % First sum each pixel across the held-out years, preserving the nine
    % target models separately. Then aggregate pixels within each cluster.
    GroupSSE = cell(nVariant,1);

    for v = 1:nVariant
        pixelSSE = squeeze(sum(squaredErrorCube{v},2));
        if nModel == 1
            pixelSSE = pixelSSE(:);
        end

        GroupSSE{v} = zeros(nGroup,nModel);
        for m = 1:nModel
            GroupSSE{v}(:,m) = accumarray( ...
                groupIndex, ...
                pixelSSE(:,m), ...
                [nGroup 1], ...
                @sum, ...
                0);
        end
    end

    % Observed full-data point estimates are invariant to cluster design.
    observed = observedOverallFromGroupSSE( ...
        GroupSSE,observationCountByGroup,comparisonNames, ...
        referenceVariant,candidateVariant);

    nComparison = numel(comparisonNames);
    BootDelta = NaN(nBootstrap,nComparison);
    BootImprovement = NaN(nBootstrap,nComparison);

    rng(seed,'twister');

    for b = 1:nBootstrap
        sampledGroup = randi(nGroup,nGroup,1);
        nObsBoot = sum(observationCountByGroup(sampledGroup));

        rmseByVariant = cell(nVariant,1);
        for v = 1:nVariant
            totalSSE = sum(GroupSSE{v}(sampledGroup,:),1);
            rmseByVariant{v} = sqrt(totalSSE ./ nObsBoot);
        end

        for ic = 1:nComparison
            refV = referenceVariant(ic);
            candV = candidateVariant(ic);

            rmseRef = rmseByVariant{refV}(:);
            rmseCand = rmseByVariant{candV}(:);

            BootDelta(b,ic) = mean(rmseCand-rmseRef);
            BootImprovement(b,ic) = ...
                mean(100*(rmseRef-rmseCand)./rmseRef);
        end
    end

    Design = repmat({designName},nComparison,1);
    BlockSizeDeg = repmat(blockSizeDeg,nComparison,1);
    NClusters = repmat(nGroup,nComparison,1);
    BootstrapReplicates = repmat(nBootstrap,nComparison,1);
    Comparison = comparisonNames(:);
    ReferenceVariant = cell(nComparison,1);
    CandidateVariant = cell(nComparison,1);
    ObservedMeanDeltaRMSE_Log = NaN(nComparison,1);
    ObservedMeanRMSEImprovementPct_Log = NaN(nComparison,1);
    DeltaRMSE_CI_Lower = NaN(nComparison,1);
    DeltaRMSE_CI_Upper = NaN(nComparison,1);
    ImprovementPct_CI_Lower = NaN(nComparison,1);
    ImprovementPct_CI_Upper = NaN(nComparison,1);
    ProbabilityCandidateBetter_RMSE = NaN(nComparison,1);
    BootstrapP_TwoSided_RMSE = NaN(nComparison,1);

    for ic = 1:nComparison
        ReferenceVariant{ic} = variantNames{referenceVariant(ic)};
        CandidateVariant{ic} = variantNames{candidateVariant(ic)};

        ObservedMeanDeltaRMSE_Log(ic) = ...
            observed.ObservedMeanDeltaRMSE_Log(ic);
        ObservedMeanRMSEImprovementPct_Log(ic) = ...
            observed.ObservedMeanRMSEImprovementPct_Log(ic);

        ci = percentileCI(BootDelta(:,ic),alpha);
        DeltaRMSE_CI_Lower(ic) = ci(1);
        DeltaRMSE_CI_Upper(ic) = ci(2);

        ci = percentileCI(BootImprovement(:,ic),alpha);
        ImprovementPct_CI_Lower(ic) = ci(1);
        ImprovementPct_CI_Upper(ic) = ci(2);

        ProbabilityCandidateBetter_RMSE(ic) = ...
            mean(BootDelta(:,ic) < 0);
        BootstrapP_TwoSided_RMSE(ic) = ...
            bootstrapTwoSidedP(BootDelta(:,ic));
    end

    T = table( ...
        Design, ...
        BlockSizeDeg, ...
        NClusters, ...
        BootstrapReplicates, ...
        Comparison, ...
        ReferenceVariant, ...
        CandidateVariant, ...
        ObservedMeanDeltaRMSE_Log, ...
        ObservedMeanRMSEImprovementPct_Log, ...
        DeltaRMSE_CI_Lower, ...
        DeltaRMSE_CI_Upper, ...
        ImprovementPct_CI_Lower, ...
        ImprovementPct_CI_Upper, ...
        ProbabilityCandidateBetter_RMSE, ...
        BootstrapP_TwoSided_RMSE);

    S = struct();
    S.Design = designName;
    S.BlockSizeDeg = blockSizeDeg;
    S.NClusters = nGroup;
    S.MeanPixelsPerCluster = mean(pixelCountByGroup);
    S.MedianPixelsPerCluster = median(pixelCountByGroup);
    S.MinPixelsPerCluster = min(pixelCountByGroup);
    S.MaxPixelsPerCluster = max(pixelCountByGroup);
end


function T = observedOverallFromGroupSSE( ...
    GroupSSE,observationCountByGroup,comparisonNames, ...
    referenceVariant,candidateVariant)

    nVariant = numel(GroupSSE);
    nModel = size(GroupSSE{1},2);
    nComparison = numel(comparisonNames);
    totalN = sum(observationCountByGroup);

    rmseByVariant = cell(nVariant,1);
    for v = 1:nVariant
        totalSSE = sum(GroupSSE{v},1);
        rmseByVariant{v} = sqrt(totalSSE ./ totalN);
    end

    Comparison = comparisonNames(:);
    ObservedMeanDeltaRMSE_Log = NaN(nComparison,1);
    ObservedMeanRMSEImprovementPct_Log = NaN(nComparison,1);

    for ic = 1:nComparison
        rmseRef = rmseByVariant{referenceVariant(ic)}(:);
        rmseCand = rmseByVariant{candidateVariant(ic)}(:);

        if numel(rmseRef) ~= nModel
            error('Unexpected target count in observed comparison.');
        end

        ObservedMeanDeltaRMSE_Log(ic) = mean(rmseCand-rmseRef);
        ObservedMeanRMSEImprovementPct_Log(ic) = ...
            mean(100*(rmseRef-rmseCand)./rmseRef);
    end

    T = table( ...
        Comparison, ...
        ObservedMeanDeltaRMSE_Log, ...
        ObservedMeanRMSEImprovementPct_Log);
end


function ci = percentileCI(x,alpha)
    x = x(isfinite(x));
    if isempty(x)
        ci = [NaN NaN];
        return
    end
    ci = prctile(x,[100*alpha/2,100*(1-alpha/2)]);
end


function p = bootstrapTwoSidedP(x)
    x = x(isfinite(x));
    if isempty(x)
        p = NaN;
        return
    end

    pLower = mean(x <= 0);
    pUpper = mean(x >= 0);
    p = min(1,2*min(pLower,pUpper));

    % Avoid reporting an exact zero merely because B is finite.
    if p == 0
        p = 2/(numel(x)+1);
    end
end


function T = convertNumericTableColumns(T,columnIndex)
    names = T.Properties.VariableNames;
    for i = columnIndex
        name = names{i};
        if iscell(T.(name))
            T.(name) = cell2mat(T.(name));
        end
    end
end


function row = structToStructureRow(S)
    row = { ...
        S.Design, ...
        S.BlockSizeDeg, ...
        S.NClusters, ...
        S.MeanPixelsPerCluster, ...
        S.MedianPixelsPerCluster, ...
        S.MinPixelsPerCluster, ...
        S.MaxPixelsPerCluster};
end
