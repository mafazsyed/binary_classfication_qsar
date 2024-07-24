% ............................. %
% k-Nearest Neighbor (k-NN) Model

% Load the QSAR_data.mat to MATLAB
load('QSAR_data.mat');

% Check for missing values in the data
if any(ismissing(QSAR_data))
    disp('The data contains missing values.')
    % Handle missing values by removing rows with missing values
    QSAR_data = rmmissing(QSAR_data);
end

% Check for and remove duplicate rows
[~, uniqueRows] = unique(QSAR_data, 'rows', 'stable');
QSAR_data = QSAR_data(uniqueRows, :);
disp(['Number of unique observations: ', num2str(size(QSAR_data, 1))]);
fprintf('\n');

% Feature scaling
features = QSAR_data(:, 1:41);
features = (features - mean(features)) ./ std(features);

% Combine data and labels for the complete dataset
X = features;
y = QSAR_data(:, 42);

% Apply PCA
[coeff, score, ~, ~, explained] = pca(X);

% Determine the number of components to keep
explainedVarianceToKeep = 95;
cumulativeExplained = cumsum(explained);
numComponents = find(cumulativeExplained >= explainedVarianceToKeep, 1);

% Keep only the selected number of components
X_pca = score(:, 1:numComponents);

% Define the grid of hyperparameters to search over
k_values = [1, 3, 5, 7, 9]; % Number of neighbors

% Initialize the best accuracy and best hyperparameters
best_accuracy = -Inf;
best_k = NaN;

% Number of folds
num_folds = 5; 

% Generate cross-validation indices

% Initialize total confusion matrix
total_conf_matrix = zeros(2, 2); % Assuming binary classification
indices = crossvalind('Kfold', y, num_folds);

% Perform a grid search over the hyperparameters
for k = k_values

	 % Initialize confusion matrices for cross-validation
	 cv_conf_matrices = zeros(2, 2, num_folds); % Assuming binary classification

    fold_accuracies = zeros(num_folds, 1);

    for i = 1:num_folds
        % Split the data into training and test sets
        test = (indices == i); 
        train = ~test;
        X_train = X_pca(train, :);
        y_train = y(train);
        X_test = X_pca(test, :);
        y_test = y(test);
        
        % Train the k-NN model
        Mdl = fitcknn(X_train, y_train, 'NumNeighbors', k);
        
        % Predict the labels for the test set
        predicted_labels_test = predict(Mdl, X_test);
        
        % Calculate the accuracy on the test set
        fold_accuracies(i) = sum(predicted_labels_test == y_test) / length(y_test);
		
		% Compute confusion matrix for this fold
        conf_matrix = confusionmat(y_test, predicted_labels_test);
        cv_conf_matrices(:, :, i) = conf_matrix;

        % Update total confusion matrix
        total_conf_matrix = total_conf_matrix + conf_matrix;
		
    end

    % Average accuracy for this k
    avg_accuracy = mean(fold_accuracies);

    % Update best model if current one is better
    if avg_accuracy > best_accuracy
        best_accuracy = avg_accuracy;
        best_k = k;
    end
end

fprintf('k-Nearest Neighbor (k-NN) Model \n');

% Display confusion matrices
for i = 1:num_folds
    fprintf('Confusion Matrix for Fold %d:\n', i);
    disp(cv_conf_matrices(:, :, i));
end

% Display the best model results
fprintf('Best Accuracy: %.2f%% with k = %d\n', best_accuracy * 100, best_k);
fprintf('Mean Accuracy: %.2f%% with k = %d\n', avg_accuracy * 100, best_k);
fprintf('Total Confusion Matrix (across all folds):\n');
disp(total_conf_matrix);

% ............................. %
% Logistic Regression Model 2

% Load the QSAR_data.mat to MATLAB 
load('QSAR_data.mat');

% Normalize features
X = normalize(QSAR_data(:, 1:end-1));
y = QSAR_data(:, end);

% Number of folds
num_folds = 5; 

% Generate cross-validation indices
indices = crossvalind('Kfold', y, num_folds);

% Initialize variables to store performance
accuracies = zeros(num_folds, 1);

% Logistic Regression parameters
max_iterations = 100;
epsilon = 1e-6;
lambda = 1e-4; % Ridge regularization parameter

% Initialize total confusion matrix
total_conf_matrix = zeros(2, 2); % Assuming binary classification

% Perform k-fold cross-validation
for i = 1:num_folds
    % Split the data into training and test sets
    test_idx = (indices == i); 
    train_idx = ~test_idx;

    X_train = X(train_idx, :);
    y_train = y(train_idx, :);
    X_test = X(test_idx, :);
    y_test = y(test_idx, :);

    % Add a column of ones for the intercept term
    X_train = [ones(size(X_train, 1), 1), X_train];
    X_test = [ones(size(X_test, 1), 1), X_test];

    % Initialize theta
    theta = zeros(size(X_train, 2), 1);

    % Logistic Regression with Newton-Raphson Method
    for iteration = 1:max_iterations
        % Logistic function
        z = X_train * theta;
        sigmoid = 1 ./ (1 + exp(-z));
    
        % Gradient
        gradient = X_train' * (y_train - sigmoid);

        % Hessian with Ridge regularization
        W = diag(sigmoid .* (1 - sigmoid));
        H = X_train' * W * X_train + lambda * eye(size(X_train, 2));

        % Update rule using pseudo-inverse for numerical stability
        delta_theta = pinv(H) * gradient;
        theta = theta + delta_theta;
    
        % Convergence check
        if norm(delta_theta) < epsilon
            break;
        end
	
    end

	% Predictions and Evaluation for this fold
	preds = 1 ./ (1 + exp(-X_test * theta)) >= 0.5;

	% Convert logical predictions to the same type as y_test (numeric)
	numeric_preds = double(preds);

	% Calculate confusion matrix for this fold
	conf_matrix = confusionmat(y_test, numeric_preds);
    
    % Update total confusion matrix
    total_conf_matrix = total_conf_matrix + conf_matrix;

    % Calculate and store accuracy for this fold
    accuracies(i) = mean(double(preds == y_test)) * 100;

end

fprintf('Logistic Regression Model 2 \n');

% Average performance across all folds
overall_accuracy = mean(accuracies);
disp(['Average Accuracy across folds: ', num2str(overall_accuracy), '%']);
% Display the total confusion matrix
fprintf('Total Confusion Matrix (across all folds):\n');
disp(total_conf_matrix);

% ............................. %
% Support Vector Machine (SVM) Model

% Load the dataset
load('QSAR_data.mat'); % Replace with your actual file path

% Assuming the data is loaded into a variable named 'QSAR_data'
features = QSAR_data(:, 1:41);
labels = QSAR_data(:, 42);

% Define the number of folds for k-fold cross-validation
k = 5;

% Create a partition for k-fold cross-validation
cv = cvpartition(size(features, 1), 'KFold', k);

% Initialize variables to accumulate performance metrics
totalClassLoss = 0;
totalAccuracy = 0;

% Initialize total confusion matrix
totalConfMatrix = zeros(2, 2); % Assuming binary classification

for i = 1:k
    % Indices for training and test set
    trainIdx = cv.training(i);
    testIdx = cv.test(i);

    % Training and test sets
    XTrain = features(trainIdx, :);
    YTrain = labels(trainIdx, :);
    XTest = features(testIdx, :);
    YTest = labels(testIdx, :);

    % Train the SVM model
    SVMModel = fitcsvm(XTrain, YTrain, 'Standardize', true, 'KernelFunction', 'RBF', 'KernelScale', 'auto');

    % Cross-validate the model
    CVSVMModel = crossval(SVMModel, 'KFold', k);

    % Calculate the classification error
    classLoss = kfoldLoss(CVSVMModel);
    totalClassLoss = totalClassLoss + classLoss;

    % Test the model
    YPred = predict(SVMModel, XTest);

    % Calculate and update total confusion matrix for this fold
    confMatrix = confusionmat(YTest, YPred);
    totalConfMatrix = totalConfMatrix + confMatrix;

    % Calculate and accumulate accuracy for this fold
    accuracy = sum(YTest == YPred) / length(YTest);
    totalAccuracy = totalAccuracy + accuracy;
end

% Calculate the average classification loss and accuracy
avgClassLoss = totalClassLoss / k;
avgAccuracy = totalAccuracy / k;

fprintf('Support Vector Machine (SVM) Model \n');

% Output the performance
fprintf('Average Classification Loss: %f\n', avgClassLoss);
fprintf('Average Test Accuracy: %f\n', avgAccuracy * 100);

% Display the total confusion matrix
fprintf('Total Confusion Matrix (across all folds):\n');
disp(totalConfMatrix);

% ............................. %
% Partial Least Squares Disciminant Analysis Model

% Load the dataset
load('QSAR_data.mat'); % Replace with your actual file path

% Preprocess the data
X = QSAR_data(:, 1:41);  % Features
y = QSAR_data(:, 42);    % Labels

% Standardize the features
[X, mu, sigma] = zscore(X);

% Set up 5-fold cross-validation
k = 5;
cv = cvpartition(size(X,1), 'KFold', k);

% Number of latent variables
numLVs = 5; % Adjust based on optimization or paper insights

% Initialize accuracy storage
accuracies = zeros(k, 1);

% Initialize total confusion matrix
total_conf_matrix = zeros(2, 2); % Assuming binary classification

for i = 1:k
    % Define training and test sets for this fold
    X_train = X(cv.training(i), :);
    y_train = y(cv.training(i), :);
    X_test = X(cv.test(i), :);
    y_test = y(cv.test(i), :);

    % Apply PLSDA
    [~, ~, ~, ~, BETA] = plsregress(X_train, y_train, numLVs);

    % Predict on test data
    y_pred = [ones(size(X_test, 1), 1) X_test] * BETA;

	% Convert predictions to binary values (0 or 1)
    y_pred_binary = y_pred > 0.5; % Using 0.5 as the threshold
    y_pred_binary = double(y_pred_binary); % Convert logical to numeric

    % Calculate and store accuracy for this fold
    accuracies(i) = sum(y_pred_binary == y_test) / numel(y_test);

    % Calculate confusion matrix for this fold
    conf_matrix = confusionmat(y_test, y_pred_binary);
    
    % Update total confusion matrix
    total_conf_matrix = total_conf_matrix + conf_matrix;

    % Calculate and store accuracy for this fold
    accuracies(i) = sum(y_pred_binary == y_test) / numel(y_test);
	
end

% Calculate average accuracy across all folds
mean_accuracy = mean(accuracies) * 100;
fprintf('Partial Least Squares Discriminant Analysis Model \n');
fprintf('Average Accuracy of PLSDA model across 5 folds: %.2f%%\n', mean_accuracy);

% Display the total confusion matrix
fprintf('Total Confusion Matrix (across all folds):\n');
disp(total_conf_matrix);