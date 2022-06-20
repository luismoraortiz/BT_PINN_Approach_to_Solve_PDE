%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Third Version of the Code                      %    by Luis Mora Ortiz 
%   Bachelor Thesis                                %    2021/2022          
%   A Deep Learning Approach to Solve Partial Differential Equations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Considerations to be made:
%   The PDE of the problem is the Heat equation
%   The BCs are Dirichlet Boundary Conditions
%   The resolution algorithm uses Sigmoid activation function
%   A finite differences routine can be employed to verify the solutions

% This code is an attempt at utilizing other different activation function
% in order to solve the same problem. This is still unsuccessful but it
% will be provided as it would be attempted to debug the code.

%% Part I.- Starting the Program and General Definitions
% Could be commented due to the large runtime of the algorithm
clear; clc; close all;
% a.   Describe the dimensions of the problem
% Set the dimensions in the x-direction
xmin = -1;
xmax =  1;
% Set the maximum time dimension
tmax =  1;
% b.   Description of the Boundary Conditions (constraints in space)
% Prescribe the number of boundary conditions at either side (temporal)
tnum = 11;
numBoundaryConditionPoints = [tnum tnum];
% Set the value of x for the Boundary Conditions
x0BC1 = xmin*ones(1,numBoundaryConditionPoints(1));
x0BC2 = xmax*ones(1,numBoundaryConditionPoints(2));
% Set the value of t for the Boundary Conditions (provided a maximum time)
t0BC1 = linspace(0,tmax,numBoundaryConditionPoints(1));
t0BC2 = linspace(0,tmax,numBoundaryConditionPoints(2));
% Set the value of the function as zero at the BC (Dirichlet BCs)
u0BC1 = zeros(1,numBoundaryConditionPoints(1));
u0BC2 = zeros(1,numBoundaryConditionPoints(2));
% c.   Description of the Initial Conditions (constraints in time)
% Prescribe the number of initial condition points (spatial)
xnum = 21;
numInitialConditionPoints  = 21;
% Set the value of x for the Initial Conditions (linearized to [-1,1])
x0IC = linspace(xmin,xmax,numInitialConditionPoints);
% Set the value of t for the Initial Conditions
t0IC = zeros(1,numInitialConditionPoints);
% Set the value of the Initial Condition as a bell curve for this scenario
alpha = 5^2;
u0IC = exp(-alpha*x0IC.^2);
% d.   Generate the final constraints by group ICs and BCs
% Write all the constraints in vector form
XBC = [x0BC1 x0BC2];
TBC = [t0BC1 t0BC2];
UBC = [u0BC1 u0BC2];
XIC = x0IC;
TIC = t0IC;
UIC = u0IC;
X0 = [x0IC x0BC1 x0BC2];
T0 = [t0IC t0BC1 t0BC2];
U0 = [u0IC u0BC1 u0BC2];
% e.   Generate the full region of internal points by using collocation.
% Set the number of points to have beyond the initial ones.
% Describing the x-coordinate:
dt = (tmax-0)/(length(t0BC1)-1);
dx = (xmax-xmin)/(length(x0IC)-1);
Xc = (xmin+dx):dx:(xmax-dx);
Tc = (0+dt):dt:(tmax-dt);
% Describing the Time matrix TT and Space matrix XX
TT = []; XX = [];
for i = 1:length(Xc)
    TT = [TT Tc'];
end
for j = 1:length(Tc)
    XX = [XX; Xc];
end
% Describing the time vector tpts and space vector xpts
tpts = []; xpts = [];
for i = 1:length(Xc)
    for j = 1:length(Tc)
        tpts = [tpts, TT(j,i)];
        xpts = [xpts, XX(j,i)];
    end
end

%% Part II.- Define the Deep Learning Model
% Let us consider a multilayer perceptron architecture which has 9 fully
% connected operations and 20 hidden neurons.
numLayers = 9;
numNeurons = 20;
% The structure will be such that it requires 2 input channels (x and t)
% and delivers a single output channel u(x,t).
XBC = [x0BC1 x0BC2];
TBC = [t0BC1 t0BC2];
UBC = [u0BC1 u0BC2];
XIC = x0IC;
TIC = t0IC;
UIC = u0IC;
X0 = [x0IC x0BC1 x0BC2];
T0 = [t0IC t0BC1 t0BC2];
U0 = [u0IC u0BC1 u0BC2];
% Inserting all of these variables into a structure that can be passed
% around easily.
inputs = struct;
inputs.XBC = XBC;
inputs.TBC = TBC;
inputs.UBC = UBC;
inputs.XIC = XIC;
inputs.TIC = TIC;
inputs.UIC = UIC;
inputs.X0 = X0;
inputs.T0 = T0;
inputs.U0 = U0;
inputs.dataX = xpts';
inputs.dataT = tpts';
parameters = generateParameters(numLayers,numNeurons);
% d.   Select the options for the fmincon algorithm
options = optimoptions('fmincon', ... % Using the multivariate restricted optimization function fmincon
    'HessianApproximation','lbfgs', ... % Using the Lim-Memory BFGS algorithm for the Hessian
    'MaxIterations',8000, ... % Needs to be sufficiently large so as to ensure convergence
    'MaxFunctionEvaluations',8000, ... % Needs to be sufficiently large so as to ensure convergence
    'OptimalityTolerance',1e-3, ... % By default considering a tolerance of 0.00001
    'SpecifyObjectiveGradient',true,... % User-defined gradient for the algorithm
    'Display','iter-detailed'); % Deliver the number of needed iterations 

% e. Implement the Model Training
parameters1 = PINN_HE_Dirichlet(inputs,parameters,options,1);
parameters2 = PINN_HE_Dirichlet(inputs,parameters,options,2);
parameters3 = PINN_HE_Dirichlet(inputs,parameters,options,3);

%% Part IV.- Evaluate and compare the characteristics of the model
% Let us first determine a set of particular time instances at which to
% study the model.
tTest = [0.00 0.10 0.25 0.50 0.75 1.00];
numPredictions = 1001;
XTest = linspace(-1,1,numPredictions);

% B. Plotting the evolution in time of a snapshot in space
figure
for i=1:numel(tTest)
    t = tTest(i);
    TTest = t*ones(1,numPredictions);
    % Make predictions.
    dlXTest = dlarray(XTest,'CB');
    dlTTest = dlarray(TTest,'CB');
    dlUPred1 = model(parameters1,dlXTest,dlTTest,1);
    dlUPred2 = model(parameters2,dlXTest,dlTTest,2);
    dlUPred3 = model(parameters3,dlXTest,dlTTest,3);
    % Calculate true values.
    alpha = 5^2; k = 0.1;
    UTest = solveHeatEquation(XTest,t,alpha,k);
    % Plot predictions.
    subplot(3,2,i)
    hold on
    plot(XTest,extractdata(dlUPred1),'-','LineWidth',2);
    plot(XTest,extractdata(dlUPred2),'-','LineWidth',2);
    plot(XTest,extractdata(dlUPred3),'-','LineWidth',2);
    ylim([-1.1, 1.1]);
    % Plot true values.
    plot(XTest, UTest, '--','LineWidth',2)
    grid minor
    hold off
end
subplot(3,2,6)
legend('Tanh','RELU','Sigmoid','Exp. Solution',...
    'Location','best','Interpreter','latex')

% C. Plotting the evolution in time of the Dirichlet boundary conditions
figure
numPredictions = 1001;
tTest = linspace(-1,1,numPredictions);
subplot(2,1,1) % Left-hand side
hold on
xTest = -1*ones(1,numPredictions);
% Make predictions for the PINN Model.
dlXTest = dlarray(xTest,'CB');
dlTTest = dlarray(tTest,'CB');
dlUPred1 = model(parameters1,dlXTest,dlTTest,1);
dlUPred2 = model(parameters2,dlXTest,dlTTest,2);
dlUPred3 = model(parameters3,dlXTest,dlTTest,3);
% Calculate true values using the Fourier series.
alpha = 5^2; k = 0.1; x = -1;
UTest = solveHeatEquation(x,tTest,alpha,k);
% Plot each of the three alternatives.
% Plot the PINN Model
plot(tTest,extractdata(dlUPred1),'-','LineWidth',2);
plot(tTest,extractdata(dlUPred2),'-','LineWidth',2);
plot(tTest,extractdata(dlUPred3),'-','LineWidth',2);
% Plot Fourier Series approximation
plot(tTest, UTest, '--','LineWidth',2)
grid minor
xlim([0,1]);
hold off

subplot(2,1,2); % Right-hand side
hold on
xTest = -1*ones(1,numPredictions);
% Make predictions for the PINN Model.
dlXTest = dlarray(xTest,'CB');
dlTTest = dlarray(tTest,'CB');
dlUPred1 = model(parameters1,dlXTest,dlTTest,1);
dlUPred2 = model(parameters2,dlXTest,dlTTest,2);
dlUPred3 = model(parameters3,dlXTest,dlTTest,3);
alpha = 5^2; k = 0.1; x = +1;
UTest = solveHeatEquation(x,tTest,alpha,k);
% Plot each of the three alternatives.
% Plot the PINN Model
plot(tTest,extractdata(dlUPred1),'-','LineWidth',2);
plot(tTest,extractdata(dlUPred2),'-','LineWidth',2);
plot(tTest,extractdata(dlUPred3),'-','LineWidth',2);
% Plot Fourier Series approximation
plot(tTest, UTest, '--','LineWidth',2)
grid minor
xlim([0,1]);
hold off
legend('Tanh','RELU','Sigmoid','Exp. Solution',...
    'Location','best','Interpreter','latex')

figure
dlU1 = [];
dlU2 = [];
dlU3 = [];
for t = 0:0.01:1
    XTest = linspace(-1,1,numPredictions);
    TTest = t*ones(1,numPredictions);
    dlXTest = dlarray(XTest,'CB');
    dlTTest = dlarray(TTest,'CB');
    dlUPred1 = model(parameters1,dlXTest,dlTTest,1);
    dlUPred2 = model(parameters2,dlXTest,dlTTest,2);
    dlUPred3 = model(parameters3,dlXTest,dlTTest,3);
    dlU1 = [dlU1, extractdata(dlUPred1)'];
    dlU2 = [dlU2, extractdata(dlUPred2)'];
    dlU3 = [dlU3, extractdata(dlUPred3)'];
end
t = 0:0.01:1;
subplot(3,1,1)
contourf(t,XTest,dlU1,1e2);
subplot(3,1,2)
contourf(t,XTest,dlU2,1e2);
subplot(3,1,3)
contourf(t,XTest,dlU3,1e2);
xlabel('Time [s]','Interpreter','latex');
ylabel('Position [m]','Interpreter','latex')
colorbar

%% Extra Functions
% 
% Solve Heat Function Equation
%
function u = solveHeatEquation(x,t,alpha,k)
% The functional form of the solution is dependent on these two values:
% \sum_i ai*sin(i pi x)*exp(-k*(i*pi)^2)
% Let us obtain each of the ai by Fourier transform
% \int_{-1}^1 exp(-alpha*x^2)*sin(i*pi*x) dx = ai * pi/2
    u = 1/2*integral(@(x) exp(-alpha*x.^2),0,1);
    for i = 1:100
        a = integral(@(x) exp(-alpha*x.^2).*cos(i*pi*x/2), 0, 1);
        % b = 2/pi*integral(@(x) exp(-alpha*x.^2).*sin(i*pi*x/2), -1, 1);
        u = u + a*exp(-k*(i*pi/2)^2.*t)*cos(i*pi.*x/2); % + b*exp(-k*(i*pi/2)^2.*t)*sin(i*pi.*x/2);
    end
end
%
% fmincon Objective Function
%
function [loss,gradientsV] = objectiveFunction(parametersV,dlX,dlT,dlXIC,dlTIC,dlUIC,dlXBC,dlTBC,dlUBC,parameterNames,parameterSizes,af)
    % Rate of Decay
    a = 0.5;
    % Convert parameters to structure of dlarray objects.
    parametersV = dlarray(parametersV);
    parameters = parameterVectorToStruct(parametersV,parameterNames,parameterSizes);
    % Evaluate model gradients and loss.
    [gradients,loss] = dlfeval(@modelGradients,parameters,dlX,dlT,dlXIC,dlTIC,dlUIC,dlXBC,dlTBC,dlUBC,af);
    % Return loss and gradients for fmincon.
    gradientsV = parameterStructToVector(gradients);
    gradientsV = a*extractdata(gradientsV);
    loss = extractdata(loss);
end
%
% Generate the Model Gradients Function
%
function [gradients,loss] = modelGradients(parameters,dlX,dlT,dlXIC,dlTIC,dlUIC,dlXBC,dlTBC,dlUBC,af)
    % Make predictions with the initial conditions.
    U = model(parameters,dlX,dlT,af);
    % Assume a value for the parameter k
    k = 0.1;
    % Calculate derivatives with respect to X and T.
    gradientsU = dlgradient(sum(U,'all'),{dlX,dlT},'EnableHigherDerivatives',true);
    Ux = gradientsU{1};
    Ut = gradientsU{2};
    % Calculate second-order derivatives with respect to X.
    Uxx = dlgradient(sum(Ux,'all'),dlX,'EnableHigherDerivatives',true);
    % Calculate lossF. Enforce Burger's equation.
    f = Ut -k.*Uxx;
    zeroTarget = zeros(size(f), 'like', f);
    lossF = mse(f, zeroTarget);
    % Calculate lossI. Enforce initial conditions.
    dlUICPred = model(parameters,dlXIC,dlTIC,af);
    lossI = mse(dlUICPred, dlUIC);
    % Calculate lossB. Enforce boundary conditions. Using Neumann BCs
    dlUBCPred = model(parameters,dlXBC,dlTBC,af);
    Un = dlgradient(sum(dlUBCPred,'all'),dlXBC,'EnableHigherDerivatives',true);
    lossB = mse(Un, dlUBC);
    % Combine losses.
    loss = lossF + lossI + lossB;
    % Calculate gradients with respect to the learnable parameters.
    gradients = dlgradient(loss,parameters);
end
%
% Generate the Model for the Full Network
%
function dlU = model(parameters,dlX,dlT,af)
    dlXT = [dlX;dlT];
    numLayers = numel(fieldnames(parameters))/2;
    % First fully connect operation.
    weights = parameters.fc1_Weights;
    bias = parameters.fc1_Bias;
    dlU = fullyconnect(dlXT,weights,bias);
    % tanh and fully connect operations for remaining layers.
    for i=2:numLayers
        name = "fc" + i;
        if (af==1)
            dlU = tanh(dlU);
        elseif (af==2)
            dlU = max(dlU,0);
        else
            dlU = 1./(1+exp(-dlU));
        end
        weights = parameters.(name + "_Weights");
        bias = parameters.(name + "_Bias");
        dlU = fullyconnect(dlU, weights, bias);
    end
end
%
% Initialize He and Initialize Zero for the Parameters
%
function parameter = initializeHe(sz,numIn,i,className)
    arguments
        sz
        numIn
        i = 0
        className = 'single'
    end
    s = rng(i);
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
%
% Parameter Struct to Vector and Vector to Struct
%
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
%
% Function to generate the parameters struct of the function
%
function parameters = generateParameters(numLayers,numNeurons)
% Let us generate a structure for the parameters of the operation so that
% we might describe a variable that depends on the manually inputted
% neurons and layers.
    parameters = struct;
% a.   Generating the parameters for the first operation.
    % The inputs are the initial values of x and t (2-dimensions) for each 
    % of the number of total neurons, therefore numNeurons
    sz = [numNeurons 2];
    parameters.fc1_Weights = initializeHe(sz,2,1);
    parameters.fc1_Bias = initializeZeros([numNeurons 1]);
% b.   Generating the parameters for the intermediate connect operations.
    % The inputs are the numNeurons and the outputs are also numNeurons
    for layerNumber=2:numLayers-1
        name = "fc"+layerNumber;
        % It will be a nxn matrix in which element i j refers to the weight of
        % element i of the previous layer to the element j of the new one
        sz = [numNeurons numNeurons];
        % The number of inputs is the number of neurons in the previous layer
        numIn = numNeurons;
        % Initializing the parameters for the connect operation
        parameters.(name + "_Weights") = initializeHe(sz,numIn,layerNumber);
        parameters.(name + "_Bias") = initializeZeros([numNeurons 1]);
    end
% c.   Generating the parameters for the last fully connect operation.
    sz = [1 numNeurons];
    % The inputs for the last connect operation are also the number of neurons
    numIn = numNeurons; 
    % Initializing the parameters for the connect operation
    parameters.("fc" + numLayers + "_Weights") = initializeHe(sz,numIn,numLayers);
    parameters.("fc" + numLayers + "_Bias") = initializeZeros([1 1]);
end

function parameters = PINN_HE_Dirichlet(in,parameters,options,af)
% Part III.- Train the Network Using fmincon
% Generate the vector with the parameters to be inserted in the algorithm.
[parametersV,parameterNames,parameterSizes] = parameterStructToVector(parameters); 
parametersV = double(extractdata(parametersV));

% Generate the arrays with the data for loss function minimization
dlX = dlarray(in.dataX','CB');
dlT = dlarray(in.dataT','CB');
dlX0 = dlarray(in.X0,'CB');
dlT0 = dlarray(in.T0,'CB');
dlU0 = dlarray(in.U0,'CB');
dlXBC = dlarray(in.XBC,'CB');
dlTBC = dlarray(in.TBC,'CB');
dlUBC = dlarray(in.UBC,'CB');
dlXIC = dlarray(in.XIC,'CB');
dlTIC = dlarray(in.TIC,'CB');
dlUIC = dlarray(in.UIC,'CB');

% Define the objective function through the data obtained beforehand
objFun = @(parameters) objectiveFunction(parameters,dlX,dlT,...
    dlXIC,dlTIC,dlUIC,dlXBC,dlTBC,dlUBC,parameterNames,parameterSizes,af);

% Run the fmincon algorithm and obtain the total runtime of the algorithm
tic
%[x,fval,exitflag,output,lambda,grad,hessian]
[parametersV,fval,exitflag,output,~,~,~] = fmincon(objFun,parametersV,[],[],[],[],[],[],[],options);
toc
% Revert the optimized version of the parameters to a struct. 
parameters = parameterVectorToStruct(parametersV,parameterNames,parameterSizes);
end
