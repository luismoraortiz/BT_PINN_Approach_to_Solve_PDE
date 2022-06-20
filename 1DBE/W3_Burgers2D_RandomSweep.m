%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Fourth Version of the Code                     %    by Luis Mora Ortiz 
%   Bachelor Thesis                                %    2021/2022          
%   A Deep Learning Approach to Solve Partial Differential Equations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Considerations to be made:
%   The PDE of the problem is the Burgers equation
%   The BCs are Dirichlet Boundary Conditions
%   The resolution algorithm uses Sigmoid activation function

% This problem will solve the 1DBE in order to compare a set of sample
% solutions and then study their performance in terms of accuracy,
% efficiency and convergence. The archive containing all the data will be
% "Data_1DBE_Sweep.mat".

%% Part I.- Starting the Program and General Definitions
% Could be commented due to the large runtime of the algorithm
clc; clear; close all;

%% Part II.- Definition of Hyperparameters and Description of the Problem
% a. Let us first provide the key hyperparameters and geometrical defs.
% Prescribe the number of boundary conditions at either side (temporal)
numBoundaryConditionPoints = [11 11];
% Set the value of x for the Boundary Conditions
x0BC1 = -1*ones(1,numBoundaryConditionPoints(1));
x0BC2 = ones(1,numBoundaryConditionPoints(2));
% Set the value of t for the Boundary Conditions (provided a maximum time)
tmax = 1;
t0BC1 = linspace(0,tmax,numBoundaryConditionPoints(1));
t0BC2 = linspace(0,tmax,numBoundaryConditionPoints(2));
% Set the value of the function as zero at the BC (Dirichlet BCs)
u0BC1 = zeros(1,numBoundaryConditionPoints(1));
u0BC2 = zeros(1,numBoundaryConditionPoints(2));
% Prescribe the number of initial condition points (spatial)
numInitialConditionPoints  = 21;
% Set the value of x for the Initial Conditions (linearized to [-1,1])
x0IC = linspace(-1,1,numInitialConditionPoints);
% Set the value of t for the Initial Conditions
t0IC = zeros(1,numInitialConditionPoints);
% Set the value of the Initial Condition as a sine wave
u0IC = -sin(x0IC*pi);
% Write all the constraints in vector form
X0 = [x0IC x0BC1 x0BC2];
T0 = [t0IC t0BC1 t0BC2];
U0 = [u0IC u0BC1 u0BC2];
% The validity of this approach relies solely in the fact that the
% prescribed value is to the function in the ICs or BCs (Dirichlet) and
% should be changed otherwise
dlX0 = dlarray(X0,'CB');
dlT0 = dlarray(T0,'CB');
dlU0 = dlarray(U0,'CB');


%% e.- Define the Deep Learning Model
% The structure will be such that it requires 2 input channels (x and t)
% and delivers a single output channel u(x,t).
% Allot the number of layers
numLayers = 12;
% Allot the number of neurons
numNeurons = 5;

% Describing the options for the optimization model.
options = optimoptions('fmincon', ... % Using the multivariate restricted optimization function fmincon
    'HessianApproximation','lbfgs', ... % Using the Lim-Memory BFGS algorithm for the Hessian
    'MaxIterations',2000, ... % Needs to be sufficiently large so as to ensure convergence
    'MaxFunctionEvaluations',5000, ... % Needs to be sufficiently large so as to ensure convergence
    'OptimalityTolerance',1e-3, ... % By default considering a tolerance of 0.00001
    'SpecifyObjectiveGradient',true); % User-defined gradient for the algorithm

%% Considering the regions at which to analyze the solution
% Testing the initial condition
numPredictions = 1001;
tTest_IC = zeros(1,numPredictions);
xTest_IC = linspace(-1,1,numPredictions);
dlXTest_IC = dlarray(xTest_IC,'CB');
dlTTest_IC = dlarray(tTest_IC,'CB');
UTest_IC = solveBurgers(xTest_IC,0,0.01/pi);
% Testing the final condition
tTest_FC = ones(1,numPredictions);
xTest_FC = linspace(-1,1,numPredictions);
dlXTest_FC = dlarray(xTest_FC,'CB');
dlTTest_FC = dlarray(tTest_FC,'CB');
UTest_FC = solveBurgers(xTest_FC,1,0.01/pi);
% Testing the left BC
tTest_LC = linspace(0,1,numPredictions);
xTest_LC = ones(1,numPredictions);
dlXTest_LC = dlarray(xTest_LC,'CB');
dlTTest_LC = dlarray(tTest_LC,'CB');
UTest_LC = zeros(1,numPredictions);
% Testing the right BC
tTest_RC = linspace(0,1,numPredictions);
xTest_RC = -ones(1,numPredictions);
dlXTest_RC = dlarray(xTest_LC,'CB');
dlTTest_RC = dlarray(tTest_LC,'CB');
UTest_RC = zeros(1,numPredictions);
% Solving for the Left and Right Boundary Conditions
for i = 1:numPredictions
    UTest_LC(i) = solveBurgers(-1,tTest_LC(i),0.01/pi);
    UTest_RC(i) = solveBurgers( 1,tTest_RC(i),0.01/pi);
end

% Initializing the efficiency metrics.
err_IC = zeros(1,100);
err_FC = zeros(1,100);
err_LC = zeros(1,100);
err_RC = zeros(1,100);
err_BC = zeros(1,100);
convs  = zeros(1,100);
ttaken = zeros(1,100);
niters = zeros(1,100);
nevals = zeros(1,100);
%% Sweeping the solution with a sample of 100 trials
for i = 1:100
    % Set the number of points to have beyond the initial ones.
    numInternalCollocationPoints = 1e4;
    % This will separate the number of collocation points in a (quasi-)uniform
    % manner on the x-t plane (two dimensional)
    pointSet = sobolset(2);
    points = net(pointSet,numInternalCollocationPoints);
    % Setting x0 = 0 as x = -1 and x0 = 1 as x = 1 (should be modified if
    % x-coordinates are modified
    dataX = 2*points(:,1)-1; 
    % Originally defined in the interval [0,1], does not need to be modified
    dataT = points(:,2);
    % Initialize a struct por the first connect op.
    parameters = struct;
    % The first connect operation has two inputs
    sz = [numNeurons 2];
    % Initialize the weights; the number of inputs will be equal to 2
    parameters.fc1_Weights = initializeHe(sz,2,'double');
    % Initialize the bias; the number of outputs will be the num of neurons
    parameters.fc1_Bias = initializeZeros([numNeurons 1],'double');
    % Initializing each of the intermediate layers
    for layerNumber=2:numLayers-1
        name = "fc"+layerNumber;
        sz = [numNeurons numNeurons];
        numIn = numNeurons;
        % The number of inputs will be equal to the number of neurons
        parameters.(name + "_Weights") = initializeHe(sz,numIn,'double');
        % The number of outputs will be the number of neurons
        parameters.(name + "_Bias") = initializeZeros([numNeurons 1],'double');
    end
    % The last connect operation has only one output channel
    sz = [1 numNeurons];
    numIn = numNeurons;
    % The number of inputs will be equal to the number of neurons
    parameters.("fc" + numLayers + "_Weights") = initializeHe(sz,numIn,'double');
    % The number of outputs will be equal to one
    parameters.("fc" + numLayers + "_Bias") = initializeZeros([1 1],'double');
    % The last connect operation has only one output channel
    sz = [1 numNeurons];
    numIn = numNeurons;
    % The number of inputs will be equal to the number of neurons
    parameters.("fc" + numLayers + "_Weights") = initializeHe(sz,numIn,'double');
    % The number of outputs will be equal to one
    parameters.("fc" + numLayers + "_Bias") = initializeZeros([1 1],'double');
    [parametersV,parameterNames,parameterSizes] = parameterStructToVector(parameters); 
    parametersV = double(extractdata(parametersV));
    dlX = dlarray(dataX','CB');
    dlT = dlarray(dataT','CB');
    % Solving the problem
    objFun = @(parameters) objectiveFunction(parameters,dlX,dlT,dlX0,dlT0,dlU0,parameterNames,parameterSizes);
    tic
    [parametersV,~,~,outputs] = fmincon(objFun,parametersV,[],[],[],[],[],[],[],options);
    ttk = toc;
    parameters = parameterVectorToStruct(parametersV,parameterNames,parameterSizes);
    % Testing the initial condition
    dlUPred = model(parameters,dlXTest_IC,dlTTest_IC);
    err_IC(i) = mse(extractdata(dlUPred),UTest_IC);
    % Testing the right condition
    dlUPred = model(parameters,dlXTest_FC,dlTTest_FC);
    err_FC(i) = mse(extractdata(dlUPred),UTest_FC);
    % Testing the left boundary condition
    dlUPred = model(parameters,dlXTest_LC,dlTTest_LC);
    err_LC(i) = mse(extractdata(dlUPred),UTest_LC);
    % Testing the right boundary condition
    dlUPred = model(parameters,dlXTest_RC,dlTTest_RC);
    err_RC(i) = mse(extractdata(dlUPred),UTest_RC);
    % Combining the boundary conditions
    err_BC(i) = (err_LC(i) + err_RC(i))/2;
    % Gauging the efficiency metrics
    ttaken(i) = ttk;
    niters(i) = outputs.iterations;
    nevals(i) = outputs.funcCount;
    convs(i) = (err_BC(i)<0.02)&(err_IC(i)<0.02);
    disp(i)
end


%% Checkmark for results: 
% if the .mat data is loaded, the archive might be opened from here and
% only the sections above will be ran.

%% Developping the plots and visualizations
convs2 = err_FC<0.02;
sum((convs==1)&(convs2==0))
sum((convs==1)&(convs2==0)&(niters>1000))
sum((convs==1)&(convs2==1))
sum((convs==1)&(convs2==1)&(niters>1000))
sum((convs==0))
sum((convs==0)&(niters>1000))
figure
% Considering the error in the Boundary Conditions and Initial Conditions
subplot(1,2,1)
hold on
scatter(err_BC((convs==1)&(convs2==0)),err_IC((convs==1)&(convs2==0)),'gs');
scatter(err_BC((convs==1)&(convs2==1)),err_IC((convs==1)&(convs2==1)),'bo');
scatter(err_BC(convs==0),err_IC(convs==0),'rx');
hold off
xlabel('Error in the Boundary Conditions','Interpreter','latex');
ylabel('Error in the Initial Conditions','Interpreter','latex');
set(gca,'YScale','log');
set(gca,'XScale','log');
grid minor
box on
% Considering the error in the Boundary Conditions and the Final Time
subplot(1,2,2)
hold on
scatter(err_BC((convs==1)&(convs2==1)),err_FC((convs==1)&(convs2==1)),'bo');
scatter(err_BC(convs==0),err_FC(convs==0),'rx');
scatter(err_BC((convs==1)&(convs2==0)),err_FC((convs==1)&(convs2==0)),'gs');
hold off
xlabel('Error in the Boundary Conditions','Interpreter','latex');
ylabel('Error in the Final Conditions','Interpreter','latex');
set(gca,'YScale','log');
set(gca,'XScale','log');
grid minor
box on
legend('True Positive','True Negative','Fake Positive','Interpreter','latex','Location','best')

%% Representing the bubble chart
figure
hold on
bubblechart(niters((convs==1)&(convs2==1)),nevals((convs==1)&(convs2==1)),ttaken((convs==1)&(convs2==1)),'b');
bubblechart(niters((convs==1)&(convs2==0)),nevals((convs==1)&(convs2==0)),ttaken((convs==1)&(convs2==0)),'g');
bubblechart(niters(convs==0),nevals(convs==0),ttaken(convs==0),'r');
hold off
%set(gca,'YScale','log');
%set(gca,'XScale','log');
grid minor
xlabel('Number of Iterations ($n_{iter}$)','Interpreter','latex');
ylabel('Number of Function Evaluations ($f_{evals}$)','Interpreter','latex');
legend('True Positive','Fake Positive','True Negative','Interpreter','latex','Location','best')
blgd = bubblelegend('Size of the ball in $t(s)$',...
    'Interpreter','latex','Location','northwest');
bubblesize([3,25])
bubblelim([1.4,1.5e3])
box on

%% Considering some interesting stats of the sample.
cc = (convs==1)&(convs2==1);
mean(err_BC(cc))
mean(err_IC(cc))
std(err_BC(cc))
std(err_IC(cc))
cov(err_IC(cc)',err_BC(cc)')
corr(err_BC(cc)',err_IC(cc)')

min(ttaken(cc)')
max(ttaken(cc)')
mean(ttaken(cc)')
mean(nevals(cc)')
mean(niters(cc)')
std(ttaken(cc)')
std(nevals(cc)')
std(niters(cc)')
corr(nevals(cc)',ttaken(cc)')
corr(nevals(cc)',niters(cc)')
corr(ttaken(cc)',niters(cc)')

%% EXTRA FUNCTIONS
function U = solveBurgers(X,t,nu)
% Define functions.
f = @(y) exp(-cos(pi*y)/(2*pi*nu));
g = @(y) exp(-(y.^2)/(4*nu*t));
% Initialize solutions.
U = zeros(size(X));
% Loop over x values.
for i = 1:numel(X)
    x = X(i);
    % Calculate the solutions using the integral function. The boundary
    % conditions in x = -1 and x = 1 are known, so leave 0 as they are
    % given by initialization of U.
    if abs(x) ~= 1
        fun = @(eta) sin(pi*(x-eta)) .* f(x-eta) .* g(eta);
        uxt = -integral(fun,-inf,inf);
        fun = @(eta) f(x-eta) .* g(eta);
        U(i) = uxt / integral(fun,-inf,inf);
    end
end
end

function [loss,gradientsV] = objectiveFunction(parametersV,dlX,dlT,dlX0,dlT0,dlU0,parameterNames,parameterSizes)
% Convert parameters to structure of dlarray objects.
parametersV = dlarray(parametersV);
parameters = parameterVectorToStruct(parametersV,parameterNames,parameterSizes);
% Evaluate model gradients and loss.
[gradients,loss] = dlfeval(@modelGradients,parameters,dlX,dlT,dlX0,dlT0,dlU0);
% Return loss and gradients for fmincon.
gradientsV = parameterStructToVector(gradients);
gradientsV = extractdata(gradientsV);
loss = extractdata(loss);
end

function [gradients,loss] = modelGradients(parameters,dlX,dlT,dlX0,dlT0,dlU0)
% Make predictions with the initial conditions.
U = model(parameters,dlX,dlT);
% Calculate derivatives with respect to X and T.
gradientsU = dlgradient(sum(U,'all'),{dlX,dlT},'EnableHigherDerivatives',true);
Ux = gradientsU{1};
Ut = gradientsU{2};
% Calculate second-order derivatives with respect to X.
Uxx = dlgradient(sum(Ux,'all'),dlX,'EnableHigherDerivatives',true);
% Calculate lossF. Enforce Burger's equation.
f = Ut + U.*Ux - (0.01./pi).*Uxx;
zeroTarget = zeros(size(f), 'like', f);
lossF = mse(f, zeroTarget);
% Calculate lossU. Enforce initial and boundary conditions.
dlU0Pred = model(parameters,dlX0,dlT0);
lossU = mse(dlU0Pred, dlU0);
% Combine losses.
loss = lossF + lossU;
% Calculate gradients with respect to the learnable parameters.
gradients = dlgradient(loss,parameters);
end

function dlU = model(parameters,dlX,dlT)
dlXT = [dlX;dlT];
numLayers = numel(fieldnames(parameters))/2;
% First fully connect operation.
weights = parameters.fc1_Weights;
bias = parameters.fc1_Bias;
dlU = fullyconnect(dlXT,weights,bias);
% tanh and fully connect operations for remaining layers.
for i=2:numLayers
    name = "fc" + i;
    dlU = tanh(dlU);
    weights = parameters.(name + "_Weights");
    bias = parameters.(name + "_Bias");
    dlU = fullyconnect(dlU, weights, bias);
end
end

function parameter = initializeHe(sz,numIn,i,className)
    arguments
        sz
        numIn
        i = 0
        className = 'single'
    end
    parameter = sqrt(2/numIn) * randn(sz,className);
    % This algorithm does not converge unless the initial definition has
    % some random arbitrary values, and thus, it might be more sensible to
    % set a common seed so that the convergence is relative to that choice
    parameter = dlarray(parameter);
end

function parameter = initializeZeros(sz,className)
    arguments
        sz
        className = 'single'
    end
    parameter = zeros(sz,className);
    parameter = dlarray(parameter);
end

function parameters = parameterVectorToStruct(parametersV,parameterNames,parameterSizes)
% parameterVectorToStruct converts a vector of parameters with specified
% names and sizes to a struct.
    parameters = struct;
    numFields = numel(parameterNames);
    count = 0;
    for i = 1:numFields
        numElements = prod(parameterSizes{i});
        parameter = parametersV(count+1:count+numElements);
        parameter = reshape(parameter,parameterSizes{i});
        parameters.(parameterNames{i}) = parameter;
        count = count + numElements;
    end
end

function [parametersV,parameterNames,parameterSizes] = parameterStructToVector(parameters)
% parameterStructToVector converts a struct of learnable parameters to a
% vector and also returns the parameter names and sizes.
    % Parameter names.
    parameterNames = fieldnames(parameters);
    % Determine parameter sizes.
    numFields = numel(parameterNames);
    parameterSizes = cell(1,numFields);
    for i = 1:numFields
        parameter = parameters.(parameterNames{i});
        parameterSizes{i} = size(parameter);
    end
    % Calculate number of elements per parameter.
    numParameterElements = cellfun(@prod,parameterSizes);
    numParamsTotal = sum(numParameterElements);
    % Construct vector
    parametersV = zeros(numParamsTotal,1,'like',parameters.(parameterNames{1}));
    count = 0;
    for i = 1:numFields
        parameter = parameters.(parameterNames{i});
        numElements = numParameterElements(i);
        parametersV(count+1:count+numElements) = parameter(:);
        count = count + numElements;
    end
end



