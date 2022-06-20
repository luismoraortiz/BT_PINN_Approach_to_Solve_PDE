%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   First Version of the Code                      %    by Luis Mora Ortiz 
%   Bachelor Thesis                                %    2021/2022          
%   A Deep Learning Approach to Solve Partial Differential Equations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Considerations to be made:
%   The PDE of the problem is the Burguers equation
%   The BCs are Dirichlet Boundary Conditions
%   The resolution algorithm uses Sigmoid activation function
%   This is inspired by MathWorks code on the page:
%   Solve Partial Differential Equation with LBFGS Method and Deep Learning

%% Part 0.- Starting the Program and General Definitions
% Could be commented due to the large runtime of the algorithm
clear; clc; close all;

%% Part I. Describing the Training Data
% a.   Describe the dimensions of the problem - boundary conditions
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
% b.  Describe the dimensiones of the problem - initial conditions
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
% c.  Describe the dimensions of the problem - interior points
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

%% Part II.- Define the Deep Learning Model
% Let us consider a multilayer perceptron architecture which has 15 fully
% connected operations and 10 hidden neurons.
% Allot the number of layers
numLayers = 15;
% Allot the number of neurons
numNeurons = 10;

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

options = optimoptions('fmincon', ... % Using the multivariate restricted optimization function fmincon
    'HessianApproximation','lbfgs', ... % Using the Lim-Memory BFGS algorithm for the Hessian
    'MaxIterations',6000, ... % Needs to be sufficiently large so as to ensure convergence
    'MaxFunctionEvaluations',6000, ... % Needs to be sufficiently large so as to ensure convergence
    'OptimalityTolerance',1e-3, ... % By default considering a tolerance of 0.00001
    'Display','iter-detailed',...
    'SpecifyObjectiveGradient',true); % User-defined gradient for the algorithm

[parametersV,parameterNames,parameterSizes] = parameterStructToVector(parameters); 
parametersV = double(extractdata(parametersV));

dlX = dlarray(dataX','CB');
dlT = dlarray(dataT','CB');
dlX0 = dlarray(X0,'CB');
dlT0 = dlarray(T0,'CB');
dlU0 = dlarray(U0,'CB');

objFun = @(parameters) objectiveFunction(parameters,dlX,dlT,dlX0,dlT0,dlU0,parameterNames,parameterSizes);

tic
parametersV = fmincon(objFun,parametersV,[],[],[],[],[],[],[],options);
toc

parameters = parameterVectorToStruct(parametersV,parameterNames,parameterSizes);

%% Adding an even number of parameters to not account for the overshoot
tTest = [0.25 0.5 0.75 1];
numPredictions = 1000;
XTest = linspace(-1,1,numPredictions);

figure(3)
for i=1:numel(tTest)
    t = tTest(i);
    TTest = t*ones(1,numPredictions);
    % Make predictions.
    dlXTest = dlarray(XTest,'CB');
    dlTTest = dlarray(TTest,'CB');
    dlUPred = model(parameters,dlXTest,dlTTest);
    % Calculate true values.
    UTest = solveBurgers(XTest,t,0.01/pi);
    % Calculate error.
    err = mse(extractdata(dlUPred),UTest) / max(UTest);
    % Plot predictions.
    subplot(2,2,i)
    plot(XTest,extractdata(dlUPred),'-','LineWidth',2);
    ylim([-1.1, 1.1])
    % Plot true values.
    hold on
    plot(XTest, UTest, '--','LineWidth',2)
    hold off
    title("t = " + t + ", Relative $L_{\infty}$ Error = " + gather(err),'Interpreter','latex');
    grid minor
end

subplot(2,2,2)
legend('PINN Prediction','Analytical Value','Interpreter','latex')

%%
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
