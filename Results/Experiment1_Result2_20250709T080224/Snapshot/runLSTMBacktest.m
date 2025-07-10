function results = runLSTMBacktest(params)
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
    
    % Random seed to replicate 
    rng(params.randomSeed, "twister");

    % ====== Load data =======
    priceData = load('energyPrices.mat','energyPrices');
    priceData = priceData.energyPrices;
    
    % Clean data
    priceData = removevars(priceData,["NYDiesel","GulfDiesel","LARegular"]);
    priceData = rmmissing(priceData);
    returnData = tick2ret(priceData);

    % Prepare Data for Training LSTM Model
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

    % Specify rows to use in the backtest (31-Dec-2015 to 2-Nov-2021).
    backtest_start_idx = find(date_vector < datetime(2016,1,1),1,'last');
    backtest_indices = backtest_start_idx:size(network_responses,1);
    
    % Specify data reserved for the backtest.
    Xbacktest = network_features(backtest_indices);
    Tbacktest = network_responses(backtest_indices,:);
    
    % Remove the backtest data.
    network_features = network_features(1:backtest_indices(1)-1);
    network_responses = network_responses(1:backtest_indices(1)-1,:);
    
    % Partition the remaining data into training and validation sets.
    rng('default');
    cv_partition = cvpartition(size(network_features,1),'HoldOut',0.2);
    
    % Training set
    Xtraining = network_features(~cv_partition.test,:);
    Ttraining = network_responses(~cv_partition.test,:);
    
    % Validation set
    Xvalidation = network_features(cv_partition.test,:);
    Tvalidation = network_responses(cv_partition.test,:);

    % Define LSTM Network Architecture
    numFeatures = width(returnData);
    
    layers_LSTM = [ ...
        sequenceInputLayer(numFeatures) 
        lstmLayer(numHiddenUnits)
        layerNormalizationLayer
        lstmLayer(numHiddenUnits)
        layerNormalizationLayer
        lstmLayer(numHiddenUnits,'OutputMode','last')
        layerNormalizationLayer
        fullyConnectedLayer(numOutputs)];
    
    options_LSTM = trainingOptions('adam', ...
        'InputDataFormats',"CTB",...
        'Plots','none', ... 
        'Verbose',0, ...
        'MaxEpochs',maxEpochs, ...
        'MiniBatchSize',128, ...
        'Shuffle','every-epoch', ...
        'ValidationData',{Xvalidation,Tvalidation}, ...
        'ValidationFrequency',50, ...
        'ValidationPatience',10, ...
        'InitialLearnRate',initialLearnRate, ...
        'GradientThreshold',1);

    % Train LSTM model
    net_LSTM = trainnet(Xtraining,Ttraining,layers_LSTM,"mse",options_LSTM);

    % Prepare Backtest Data
    backtestPred_LSTM = minibatchpredict(net_LSTM,Xbacktest,InputDataFormats="CTB");
    backtestSignalTT = timetable(date_vector(backtest_indices),backtestPred_LSTM);
    backtestPriceTT = priceData(date_vector(backtest_indices),outputVarName);
    risk_free_rate = 0.01;
    
    % Create Backtest Strategies
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
    [~,idx] = max(signalVec);
    weights = zeros(size(signalVec));
    weights(idx) = 1;
end