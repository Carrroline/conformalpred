% =========================================================================
% Full-Scale Conformal Prediction Coverage Analysis
% Objective: To calculate and compare the coverage rates of In-Sample and 
% Out-of-Sample on the entire backtest dataset.
% =========================================================================

% --- 1. Hyper parameter ---
params.alpha = 0.10;           % signifiance = 1 - alpha = 90%
params.trainWin = 252;         % 1-yr
params.calWin = 60;            % ~3 months
params.retrainFreq = 21;       % retrain monthly
params.numHiddenUnits = 50;
params.maxEpochs = 500;
params.initialLearnRate = 1e-3;
params.randomSeed = 1;
rng(params.randomSeed, "twister");

fprintf('--- Full-Scale Analysis Setup ---\n');
fprintf('Target Confidence Level (1-alpha): %.2f%%\n', (1-params.alpha)*100);

% --- 2. Data Load ---
priceData = load('energyPrices.mat','energyPrices');
priceData = priceData.energyPrices;
priceData = removevars(priceData,["NYDiesel","GulfDiesel","LARegular"]);
priceData = rmmissing(priceData);
returnData = tick2ret(priceData);

historySize = 30;
futureSize = 1;
outputVarName = ["Brent" "NaturalGas", "Propane" "Kerosene"];
numOutputs = numel(outputVarName);
numFeatures = width(returnData);

start_idx = historySize + 1;
end_idx   = height(returnData) - futureSize + 1;
numSamples = end_idx - start_idx + 1;
date_vector = returnData.Time(start_idx-1:end_idx-1);

all_features  = cell(numSamples,1);
all_responses = zeros(numSamples,numOutputs);

for j = 1:numSamples
    all_features{j} = (returnData(j:j+historySize-1,:).Variables)';
    all_responses(j,:) = (returnData(j+historySize:j+historySize+futureSize-1,outputVarName).Variables)';
end

% --- 3. Analysis ---
backtest_start_idx = find(date_vector < datetime(2016,1,1),1,'last');
backtest_end_idx = numSamples;
num_total_steps = backtest_end_idx - backtest_start_idx + 1;

% (Out-of-Sample)
oos_coverage_hits = zeros(num_total_steps, numOutputs); 
oos_interval_widths = zeros(num_total_steps, numOutputs);

% (In-Sample)
is_coverage_hits_history = []; 
is_total_obs_history = [];    

net_LSTM = []; 
q_hat = [];

fprintf('--- Starting Full Analysis Loop for %d Timesteps ---\n', num_total_steps);

for i = 1:num_total_steps
    
    t = backtest_start_idx + i - 1;
    
    % --- periodically retrain ---
    if i == 1 || mod(i - 1, params.retrainFreq) == 0
        fprintf('Day %d of %d: --- RETRAINING MODEL & RECALIBRATING ---\n', i, num_total_steps);
        
        train_indices = (t - params.calWin - params.trainWin) : (t - params.calWin - 1);
        cal_indices   = (t - params.calWin) : (t - 1);
        Xtraining = all_features(train_indices);
        Ttraining = all_responses(train_indices,:);
        Xcal = all_features(cal_indices);
        Tcal = all_responses(cal_indices,:);

        layers_LSTM = [ ...
            sequenceInputLayer(numFeatures)
            lstmLayer(params.numHiddenUnits)
            layerNormalizationLayer
            lstmLayer(params.numHiddenUnits)
            layerNormalizationLayer
            lstmLayer(params.numHiddenUnits, 'OutputMode', 'last')
            layerNormalizationLayer
            fullyConnectedLayer(numOutputs)];
        
        options_LSTM = trainingOptions('adam', 'InputDataFormats',"CTB", ...
            'Plots','none', 'Verbose',0, 'MaxEpochs',params.maxEpochs, ...
            'InitialLearnRate',params.initialLearnRate, 'GradientThreshold',1);
        
        net_LSTM = trainnet(Xtraining, Ttraining, layers_LSTM, "mse", options_LSTM);
        
        pred_cal = minibatchpredict(net_LSTM, Xcal, 'InputDataFormats', 'CTB');
        conformity_scores = abs(Tcal - pred_cal);
        q_hat = quantile(conformity_scores, 1 - params.alpha);
        
        % --- In-Sample coverage rate ---
        is_lower_bound = pred_cal - q_hat;
        is_upper_bound = pred_cal + q_hat;
        is_covered = (Tcal >= is_lower_bound) & (Tcal <= is_upper_bound);
        is_coverage_hits_history = [is_coverage_hits_history; sum(is_covered, 'all')];
        is_total_obs_history = [is_total_obs_history; numel(Tcal)];
    end
    
    % --- Out-of-Sample predict ---
    Xtest = all_features{t};
    actual_y = all_responses(t, :);
    pred_test = minibatchpredict(net_LSTM, {Xtest}, 'InputDataFormats', 'CTB');
    
    lower_bound = pred_test - q_hat;
    upper_bound = pred_test + q_hat;
    
    is_covered_oos = (actual_y >= lower_bound) & (actual_y <= upper_bound);
    oos_coverage_hits(i, :) = is_covered_oos;
    oos_interval_widths(i, :) = 2 * q_hat;
    
    if mod(i, 100) == 0
        fprintf('...processed %d days.\n', i);
    end
end

% (In-Sample)
total_is_hits = sum(is_coverage_hits_history);
total_is_obs = sum(is_total_obs_history);
is_empirical_coverage = total_is_hits / total_is_obs;

% (Out-of-Sample)
total_oos_hits = sum(oos_coverage_hits, 'all');
total_oos_obs = numel(oos_coverage_hits);
oos_empirical_coverage = total_oos_hits / total_oos_obs;
avg_oos_interval_width = mean(oos_interval_widths, 'all');

fprintf('\n\n--- FULL COVERAGE ANALYSIS SUMMARY ---\n\n');
fprintf('Target Coverage Rate (1-alpha): %.2f%%\n', (1-params.alpha) * 100);
fprintf('--------------------------------------------------\n');
fprintf('IN-SAMPLE (Calibration Set) RESULTS:\n');
fprintf('  - Empirical Coverage: %.2f%% \n', is_empirical_coverage * 100);
fprintf('  - This should be very close to the target.\n\n');

fprintf('OUT-OF-SAMPLE (Test Set) RESULTS:\n');
fprintf('  - Empirical Coverage: %.2f%% \n', oos_empirical_coverage * 100);
fprintf('  - Average Interval Width: %.4f \n', avg_oos_interval_width);
fprintf('--------------------------------------------------\n');