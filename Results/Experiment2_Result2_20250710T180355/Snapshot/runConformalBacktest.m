function results = runConformalBacktest(params)
    % ====== Parameter Define =======
    % Inputs:
    %   params -
    %            .alpha: Significance level for conformal prediction.
    %            .trainWin: Lookback window for training the model.
    %            .calWin: Lookback window for the calibration set.
    %            .numHiddenUnits: Number of hidden units in the LSTM layers.
    %            .maxEpochs: Maximum number of epochs for training.
    %            .initialLearnRate: Initial learning rate for the optimizer.
    %            .randomSeed: Seed for the random number generator for reproducibility.
    %            .retrainFreq: Frequency of model retraining (unit: days)
    %
    % Outputs:
    %   results - A struct containing the key performance indicators (KPIs)
    %             of the backtest to be logged by the Experiment Manager.

    alpha = params.alpha;
    trainWin = params.trainWin; % Training window size
    calWin = params.calWin;     % Calibration window size
    
    numHiddenUnits = params.numHiddenUnits;
    maxEpochs = params.maxEpochs;
    initialLearnRate = params.initialLearnRate;

    retrainFreq = params.retrainFreq; 
    
    % Random seed to replicate 
    rng(params.randomSeed, "twister");

    % ====== Load data =======
    priceData = load('energyPrices.mat','energyPrices');
    priceData = priceData.energyPrices;
    
    % Clean data
    priceData = removevars(priceData,["NYDiesel","GulfDiesel","LARegular"]);
    priceData = rmmissing(priceData);
    returnData = tick2ret(priceData);

    % ====== Prepare Data for Training LSTM Model ====== 
    % Model is trained using a 30-day rolling window to predict 1 day in the future.
    historySize = 30;
    futureSize = 1;
    
    % Model predicts returns for oil, natural gas, propane, and kerosene.
    outputVarName = ["Brent" "NaturalGas", "Propane" "Kerosene"];
    numOutputs = numel(outputVarName);
    
    % start_idx and end_idx are the index positions in the returnData
    % timetable corresponding to the first and last date for making a prediction.
    start_idx = historySize + 1;
    end_idx   = height(returnData) - futureSize + 1;
    numSamples = end_idx - start_idx + 1;
    
    % The date_vector variable stores the dates for making predictions.
    date_vector = returnData.Time(start_idx-1:end_idx-1);

    network_features  = cell(numSamples,1);
    network_responses = zeros(numSamples,numOutputs);
    
    for j = 1:numSamples
        network_features{j} = (returnData(j:j+historySize-1,:).Variables)';
        network_responses(j,:) = ...
            (returnData(j+historySize:j+historySize+futureSize-1,outputVarName).Variables)';
    end

    % ======= Walk-Forward Loop =======
    % Specify rows to use in the backtest
    backtest_start_idx = find(date_vector < datetime(2016,1,1),1,'last');
    backtest_end_idx = numSamples;
    
    % Initialize model and signal
    final_signals = zeros(backtest_end_idx - backtest_start_idx + 1, numOutputs);
    net_LSTM = []; 
    q_hat = [];

    for t = backtest_start_idx : backtest_end_idx
        current_loop_idx = t - backtest_start_idx + 1;
        
        % Check if today is a "retraining day"
        if current_loop_idx == 1 || mod(current_loop_idx - 1, retrainFreq) == 0
            
            % Dynamic Data Slicing
            train_indices = (t - calWin - trainWin) : (t - calWin - 1);
            cal_indices   = (t - calWin) : (t - 1);
            
            Xtraining = network_features(train_indices);
            Ttraining = network_responses(train_indices,:);
            Xcal = network_features(cal_indices);
            Tcal = network_responses(cal_indices,:);
            
            numFeatures = width(returnData);
            layers_LSTM = [ ...
                sequenceInputLayer(numFeatures)
                lstmLayer(numHiddenUnits)
                layerNormalizationLayer
                lstmLayer(numHiddenUnits)
                layerNormalizationLayer
                lstmLayer(numHiddenUnits, 'OutputMode', 'last')
                layerNormalizationLayer
                fullyConnectedLayer(numOutputs)];

            options_LSTM = trainingOptions('adam', 'InputDataFormats',"CTB", ...
                'Plots','none', 'Verbose',0, 'MaxEpochs',maxEpochs, ...
                'MiniBatchSize',128, 'Shuffle','every-epoch', ...
                'InitialLearnRate',initialLearnRate, 'GradientThreshold',1);
            
            % Periodically retrain the model
            net_LSTM = trainnet(Xtraining, Ttraining, layers_LSTM, "mse", options_LSTM);
            
            % Periodically recalibrate and calculate q_hat
            pred_cal = minibatchpredict(net_LSTM, Xcal, 'InputDataFormats', 'CTB');
            conformity_scores = abs(Tcal - pred_cal); % Score function can be defined
            q_hat = quantile(conformity_scores, 1 - alpha);
        end

        % Generate a signal for the current day 
        % Use the latest trained model (net_LSTM) and the latest calculated quantile (q_hat)
        Xtest = network_features{t};
        Xtest_cell = {Xtest};
        pred_test = minibatchpredict(net_LSTM, Xtest_cell, 'InputDataFormats', 'CTB');

        % Filter the signal using conformal prediction intervals
        interval_lower_bound = pred_test - q_hat;
        interval_upper_bound = pred_test + q_hat;
        new_signal = pred_test;

        % Filter out signals with insufficient confidence
        uncertain_indices = (interval_lower_bound <= 0) & (interval_upper_bound >= 0); 
        new_signal(uncertain_indices) = 0;
        
        % Store signals
        final_signals(current_loop_idx, :) = new_signal;
    end

    % ===== Prepare Backtest Data =====
    backtest_dates = date_vector(backtest_start_idx:backtest_end_idx);
    backtestSignalTT = timetable(backtest_dates, final_signals);
    backtestPriceTT = priceData(backtest_dates,outputVarName);
    
    risk_free_rate = 0.01;
    
    % ===== Create Backtest Strategies =====
    % Specify 10 basis points as the trading cost.
    tradingCosts = 0.001;
    
    % Invest in long positions proportionally to their predicted return.
    LongStrategy = backtestStrategy('LongOnly',@LongOnlyRebalanceFcn, ...
        'TransactionCosts',tradingCosts, ...
        'LookbackWindow',1);
    
    % Invest in both long and short positions proportionally to their predicted returns.
    LongShortStrategy = backtestStrategy('LongShort',@LongShortRebalanceFcn, ...
        'TransactionCosts',tradingCosts, ...
        'LookbackWindow',1);
    
    % Invest 100% of capital into single asset with highest predicted returns.
    BestBetStrategy = backtestStrategy('BestBet',@BestBetRebalanceFcn, ...
        'TransactionCosts',tradingCosts, ...
        'LookbackWindow',1);
    
    % For comparison, invest in an equal-weighted (buy low and sell high) strategy.
    equalWeightFcn = @(w,p,s) ones(size(w)) / numel(w);
    EqualWeightStrategy = backtestStrategy('EqualWeight',equalWeightFcn, ...
        'TransactionCosts',tradingCosts, ...
        'LookbackWindow',0);

    strategies = [LongStrategy LongShortStrategy BestBetStrategy EqualWeightStrategy];

    bt = backtestEngine(strategies,'RiskFreeRate',risk_free_rate);
    bt = runBacktest(bt,backtestPriceTT,backtestSignalTT);
    summaryTable = summary(bt);

    % Package all key metrics into the 'results' struct for the Experiment Manager to log.
    results.LongOnly_TotalReturn = summaryTable{'TotalReturn', "LongOnly"};
    results.LongOnly_SharpeRatio = summaryTable{'SharpeRatio', "LongOnly"};
    results.LongOnly_MaxDrawdown = summaryTable{'MaxDrawdown', "LongOnly"};
    
    results.LongShort_TotalReturn = summaryTable{'TotalReturn', "LongShort"};
    results.LongShort_SharpeRatio = summaryTable{'SharpeRatio', "LongShort"};
    results.LongShort_MaxDrawdown = summaryTable{'MaxDrawdown', "LongShort"};
    
    results.BestBet_TotalReturn = summaryTable{'TotalReturn', "BestBet"};
    results.BestBet_SharpeRatio = summaryTable{'SharpeRatio', "BestBet"};
    results.BestBet_MaxDrawdown = summaryTable{'MaxDrawdown', "BestBet"};
    
    results.summary = summaryTable;
end


function weights = LongOnlyRebalanceFcn(~,~,signal)
    posSignal = signal.Variables;
    posSignal(posSignal < 0) = 0;
    
    if sum(posSignal) > 0
        weights = posSignal/sum(posSignal);
    else
        weights = zeros(size(posSignal));
    end
end

function weights = LongShortRebalanceFcn(~,~,signal)
    signalVec = signal.Variables;
    if sum(abs(signalVec)) > 0
        weights = signalVec/sum(abs(signalVec));
    else
        weights = zeros(size(signalVec));
    end
end

function weights = BestBetRebalanceFcn(~,~,signal)
    signalVec = signal.Variables;

    [max_signal, idx] = max(signalVec);
    if max_signal > 0
        weights = zeros(size(signalVec));
        weights(idx) = 1;
    else
        weights = zeros(size(signalVec));
    end
end