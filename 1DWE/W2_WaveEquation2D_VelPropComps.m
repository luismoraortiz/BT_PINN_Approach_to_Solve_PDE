%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Fourth Version of the Code                     %    by Luis Mora Ortiz 
%   Bachelor Thesis                                %    2021/2022          
%   A Deep Learning Approach to Solve Partial Differential Equations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Considerations to be made:
%   The PDE of the problem is the Wave equation
%   The BCs are Dirichlet Boundary Conditions
%   The resolution algorithm uses Sigmoid activation function

% This code corresponds to the study of the velocity of propagation c for
% the one-dimensional wave equation with Dirichlet Boundary Conditions. The
% archive containing all the data used will be "Data_WE1D_Performance.mat".

%% Part I.- Starting the Program and General Definitions
% Could be commented due to the large runtime of the algorithm
clear; clc; close all;

% Let us first insert a section in which the variables to be changed are
% described beforehand, pertaining mainly to the number of layers, number
% of neurons, and the number of datapoints inserted in the algorithm:

%% Part II.- Definition of Hyperparameters and Description of the Problem
% a. Let us first provide the key hyperparameters and geometrical defs.
% Prescribe the number of boundary conditions at either side (temporal)
tnum = 21;
% Prescribe the number of initial condition points (spatial)
xnum = 41;
% Set the number of points to have beyond the initial ones.
numICPs = 1e3;
% Let us consider a multilayer perceptron architecture which has 9 fully
% connected operations and 20 hidden neurons.
numLayers = 15;
numNeurons = 10;
% Also as auxiliary variables, the following constraints of fmincon.
% Set the maximum number of iterations before convergence.
numiter = 2000;
% Set the maximum number of function evaluations before convergence.
numevs = 2000;
% Set the maximum level of tolerance under which a solution is accepted.
maxtol = 1e-3;
% Describe the dimensions of the problem
% Set the dimensions in the x-direction
xmin = -1;
xmax =  1;
% Set the maximum time dimension
tmax =  1;

% b. Description of the Boundary Conditions (constraints in space)
% Prescribe the number of boundary conditions at either side (temporal)
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

% c. Description of the Initial Conditions (constraints in time)
% Prescribe the number of initial condition points (spatial)
numInitialConditionPoints  = xnum;
% Set the value of x for the Initial Conditions (linearized to [-1,1])
x0IC = linspace(xmin,xmax,numInitialConditionPoints);
% Set the value of t for the Initial Conditions
t0IC = zeros(1,numInitialConditionPoints);
% Set the value of the Initial Condition as a bell curve for this scenario
alpha = 5^2;
u0IC = exp(-alpha*x0IC.^2);

% d. Generate the final constraints by group ICs and BCs
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

%% Part II.- Define the Deep Learning Model
% The structure will be such that it requires 2 input channels (x and t)
% and delivers a single output channel u(x,t).
% This will be predefined as the boundary conditions will not be modified
% in any of the scenarios proposed.
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

% d.   Select the options for the fmincon algorithm. Could add the
% iter-detailed variable to get further insight into the results.
options = optimoptions('fmincon', ... % Using the multivariate restricted optimization function fmincon
    'HessianApproximation','lbfgs', ... % Using the Lim-Memory BFGS algorithm for the Hessian
    'MaxIterations',numiter, ... % Needs to be sufficiently large so as to ensure convergence
    'MaxFunctionEvaluations',numevs, ... % Needs to be sufficiently large so as to ensure convergence
    'OptimalityTolerance',maxtol, ... % By default considering a tolerance of 0.00001
    'SpecifyObjectiveGradient',true); % User-defined gradient for the algorithm

%% Part III.- Solve the problem for comparison
% Let us first determine a set of particular time instances at which to
% study the model, using the alternative approach.
numPredictions = 1001;
tTestInit = 0.00;
TTestInit = tTestInit*ones(1,numPredictions);
tTestFinal = 1.00;
TTestFinal = tTestFinal*ones(1,numPredictions);
XTest = linspace(-1,1,numPredictions);
dlXTest = dlarray(XTest,'CB');
dlTTestInit = dlarray(TTestInit,'CB');
dlTTestFinal = dlarray(TTestFinal,'CB');
dlUPred = struct;

XLeft = -1*ones(1,numPredictions);
TSides = linspace(0,1,numPredictions);
XRight = 1*ones(1,numPredictions);
dlXLeft = dlarray(XLeft,'CB');
dlTSides = dlarray(TSides,'CB');
dlXRight = dlarray(XRight,'CB');

% Developping a struct for the predictions
error_model_A = struct; error_model_B = struct;
error_model_C = struct; error_model_D = struct;

% Developping a struct for the outputs and a matrix of exit flags
outputs_all = struct;
flags_A = []; flags_B = []; flags_C = []; flags_D = []; ttk = struct;

% Initializing the seeds
seed = 0;

% e.   Generate the full region of internal points by using collocation.
% This will be accomplished by a set of 4 different means, which will be
% solved individually.

%%   A. True Random Meshing: randomly selecting n points within the range
% Set the number of points to have beyond the initial ones.
numInternalCollocationPoints = numICPs;
% Implementing the loop
c = logspace(-4,0,25);
for i = 1:25
    u_exp_0 = solveWaveEquation(XTest,tTestInit,5^2,c(i));
    u_exp_1 = solveWaveEquation(XTest,tTestFinal,5^2,c(i));
    for randomICPs = 16:25
        % Describing the random data internal points
        points = rand(numInternalCollocationPoints,2);
        dataX = xmin + (xmax-xmin)*points(:,1);
        dataT = tmax*points(:,2);
        inputs.dataX = dataX;
        inputs.dataT = dataT;
        % Implementing the solution of the model
        % Generate the number of parameters
        [parameters,seed] = generateParameters(numLayers,numNeurons,seed);
        % Implement the training of the model for the parameters
        tic
        [parameters,outputs,exitflag] = PINN_HE_Dirichlet(inputs,parameters,options,c(i));
        ttk.("c_" + i + "_TR_" + randomICPs) = toc;
        outputs_all.("c_" + i + "_TR_" + randomICPs) = outputs;
        flags_A = [flags_A,exitflag];
        % Solving for the model according to the testing
        dlUU = model(parameters,dlXTest,dlTTestInit);
        dlUPred_ICPs_A0.("c_" + i + "_TR_" + randomICPs) = dlUU;
        % Obtaining the associated mean squared error
        error_model_A.("c_" + i + "_TR_" + randomICPs + "_Init") = mse(dlUU,u_exp_0);
        % Solving for the model according to the testing
        dlUU = model(parameters,dlXTest,dlTTestFinal);
        dlUPred_ICPs_A1.("c_" + i + "_TR_" + randomICPs) = dlUU;
        % Obtaining the associated mean squared error
        error_model_A.("c_" + i + "_TR_" + randomICPs) = mse(dlUU,u_exp_1);
        disp("c_" + i + "_TR_" + randomICPs);
        % Solving the model on the left
        dlUU = model(parameters,dlXLeft,dlTSides);
        error_model_A.("c_" + i + "_TR_" + randomICPs + "_Left") = mse(dlUU,0*dlUU);
        dlUPred_ICPs_Left.("c_" + i + "_TR_" + randomICPs) = dlUU;
        dlUU = model(parameters,dlXRight,dlTSides);
        error_model_A.("c_" + i + "_TR_" + randomICPs + "Right") = mse(dlUU,0*dlUU);
        dlUPred_ICPs_Right.("c_" + i + "_TR_" + randomICPs) = dlUU;
    end
end

%% Checkmark for results: 
% if the .mat data is loaded, the archive might be opened from here and
% only the sections above will be ran.

%% Part IV.- Evaluate and compare the characteristics of the model
numSamples = 25;
ttks = zeros(1,25);
errIC = zeros(1,25);
errBC = zeros(1,25);
cntr = zeros(1,25);
iters = zeros(1,25);
for i = 1:numSamples
    for j = 1:25
        eemIC = error_model_A.("c_" + j + "_TR_" + i + "_Init");
        eemBC = error_model_A.("c_" + j + "_TR_" + i + "_Left") + error_model_A.("c_" + j + "_TR_" + i + "Right");
        if (eemIC<0.03)&&(eemBC<0.03)
            ttks(j) = ttks(j) + ttk.("c_" + j + "_TR_" + i);
            iters(j) = iters(j) + outputs_all.("c_" + i + "_TR_" + randomICPs).iterations;
            errIC(j) = errIC(j) + error_model_A.("c_" + j + "_TR_" + i + "_Init");
            errBC(j) = errBC(j) + error_model_A.("c_" + j + "_TR_" + i + "_Left") + ...
                error_model_A.("c_" + j + "_TR_" + i + "Right");
        else
            cntr(j) = cntr(j) + 1;
        end
    end
end
ttks = ttks./(numSamples-cntr(j));
errIC = errIC./(numSamples-cntr(j));
errBC = 0.5*errBC./(numSamples-cntr(j));

figure
subplot(2,1,1)
plot(c,ttks,'r')
set(gca,'XScale','log')
grid minor
xlabel('Value of the Propagation Velocity $c^2$','Interpreter','latex');
ylabel('Mean Time Elapsed Till Convergence [s]','Interpreter','latex');
subplot(2,1,2)
plot(c,iters,'r')
set(gca,'XScale','log')
grid minor
xlabel('Value of the Propagation Velocity $c^2$','Interpreter','latex');
ylabel('Number of Iterations Till Convergence [-]','Interpreter','latex');


figure
subplot(2,1,1)
plot(c,errIC,'r')
set(gca,'XScale','log')
grid minor
xlabel('Value of the Propagation Velocity $c^2$','Interpreter','latex');
ylabel('Mean Squared Error on the ICs','Interpreter','latex');
subplot(2,1,2)
plot(c,errBC,'r')
set(gca,'XScale','log')
grid minor
xlabel('Value of the Propagation Velocity $c^2$','Interpreter','latex');
ylabel('Mean Squared Error on the BCs','Interpreter','latex');
    


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

% The Fourier expansion approach that was used in the HE has not been
% reproduced in the context of this code.
function u = solveWaveEquation(x,t,alpha,c)
% The functional form of the solution is dependent on these two values:
% \sum_i ai*sin(i pi x)*exp(-k*(i*pi)^2)
% Let us obtain each of the ai by Fourier transform
% \int_1^1 exp(-alpha*x^2)*sin(i*pi*x) dx = ai * pi/2
c = sqrt(c);
u = 1/2*integral(@(x) exp(-alpha*x.^2),0,1);
for i = 1:100
    a = integral(@(x) exp(-alpha*x.^2).*cos(i*pi*x/2), 0, 1);
    u = u + a*cos(c*i*pi/2.*t)*cos(i*pi/2.*x);
end
end
%
% fmincon Objective Function
%
function [loss,gradientsV] = objectiveFunction(parametersV,dlX,dlT,dlXIC,dlTIC,dlUIC,dlXBC,dlTBC,dlUBC,parameterNames,parameterSizes,c)
    % Rate of Decay
    a = 0.5;
    % Convert parameters to structure of dlarray objects.
    parametersV = dlarray(parametersV);
    parameters = parameterVectorToStruct(parametersV,parameterNames,parameterSizes);
    % Evaluate model gradients and loss.
    [gradients,loss] = dlfeval(@modelGradients,parameters,dlX,dlT,dlXIC,dlTIC,dlUIC,dlXBC,dlTBC,dlUBC,c);
    % Return loss and gradients for fmincon.
    gradientsV = parameterStructToVector(gradients);
    gradientsV = a*extractdata(gradientsV);
    loss = extractdata(loss);
end
%
% Generate the Model Gradients Function
%
function [gradients,loss] = modelGradients(parameters,dlX,dlT,dlXIC,dlTIC,dlUIC,dlXBC,dlTBC,dlUBC,k)
    % Make predictions with the initial conditions.
    U = model(parameters,dlX,dlT);
    % Assume a value for the parameter k
    % k = 0.1;
    % Calculate derivatives with respect to X and T.
    gradientsU = dlgradient(sum(U,'all'),{dlX,dlT},'EnableHigherDerivatives',true);
    Ux = gradientsU{1};
    Ut = gradientsU{2};
    % Calculate second-order derivatives with respect to X.
    Uxx = dlgradient(sum(Ux,'all'),dlX,'EnableHigherDerivatives',true);
    % Calculate second-order derivatives with respect to T.
    Utt = dlgradient(sum(Ut,'all'),dlT,'EnableHigherDerivatives',true);
    % Calculate lossF. Enforce Burger's equation.
    f = Utt -k.*Uxx;
    zeroTarget = zeros(size(f), 'like', f);
    lossF = mse(f, zeroTarget);
    % Calculate lossI. Enforce initial conditions.
    dlUICPred = model(parameters,dlXIC,dlTIC);
    lossI = mse(dlUICPred, dlUIC);
    % Calculate lossV. Loss in the derivative of the initial conditions.
    dlVICPred = model(parameters,dlXIC,dlTIC);
    Un = dlgradient(sum(dlVICPred,'all'),dlTIC,'EnableHigherDerivatives',true);
    lossV = mse(Un,0*Un);
    % Calculate lossB. Enforce boundary conditions. Using Dirichlet BCs
    dlUBCPred = model(parameters,dlXBC,dlTBC);
    Un = dlgradient(sum(dlUBCPred,'all'),dlXBC,'EnableHigherDerivatives',true);
    lossB = mse(dlUBCPred, dlUBC);
    % Combine losses.
    loss = lossF + lossI + lossV + lossB;
    % Calculate gradients with respect to the learnable parameters.
    gradients = dlgradient(loss,parameters);
end
%
% Generate the Model for the Full Network
%
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
%
% Initialize He and Initialize Zero for the Parameters
%
function parameter = initializeHe(sz,numIn,i,className)
    arguments
        sz
        numIn
        i = 0;
        className = 'single'
    end
    s = rng(i);
    parameter = sqrt(2/numIn) * randn(sz,className);
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
% Alternate Finite Differences Routine
%
% function u = findiff(x,t,alpha) 
%     % Problem Initialization
%     % Describing the initial condition and boundary conditions
%     phi0 = exp(-alpha*x.^2);
%     phivar = phi0;
%     dt = diff(t); dx = diff(x);
%     % Applying the resolution of the problem
%     beta = 0.1;
%     sz = length(x);
%     r = beta*dt(1)/(dx(1)^2); % for stability, must be 0.5 or less
%     for j = 2:length(t) % for time steps
%         phi = phi0(); 
%         M = diag(-2*ones(1,sz)) + diag(ones(1,sz-1),1) + diag(ones(1,sz-1),-1);
%         phi = phi + r*M\phi';
%         phi0 = phi;
%         phivar = [phivar;phi0];
%     end
%     u = phivar;
% end
% This is the commented original version of the finite differences
% algorithm, which did not account for a more efficient matrix
% implementation.
function u = findiff(x,t,alpha) 
    % Problem Initialization
    % Describing the initial condition and boundary conditions
    phi0 = exp(-alpha*x.^2);
    phiL = 0;
    % phiR = 0;
    phivar = zeros(length(t),length(x));
    phivar(1,:) = phi0;
    dt = diff(t); dx = diff(x);
    % Applying the resolution of the problem
    beta = 0.1;
    r = beta*dt(1)/(dx(1)^2); % for stability, must be 0.5 or less
    for j = 2:length(t) % for time steps
        phi = phi0();        
        for i = 1:length(x) % for space steps
            if i == 1 || i == length(x)
                phi(i) = phiL;
            else
                phi(i) = phi(i)+r*(phi(i+1)-2*phi(i)+phi(i-1));
            end
        end
        phi0 = phi;
        phivar(j,:) = [phivar;phi0];
    end
    u = phivar;
end
%
% Function to generate the parameters struct of the function
%
function [parameters,seed] = generateParameters(numLayers,numNeurons,seed)
% Let us generate a structure for the parameters of the operation so that
% we might describe a variable that depends on the manually inputted
% neurons and layers.
    parameters = struct;
% a.   Generating the parameters for the first operation.
    % The inputs are the initial values of x and t (2-dimensions) for each 
    % of the number of total neurons, therefore numNeurons
    sz = [numNeurons 2];
    parameters.fc1_Weights = initializeHe(sz,2,seed);
    seed = seed + 1;
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
        parameters.(name + "_Weights") = initializeHe(sz,numIn,seed);
        seed = seed + 1;
        parameters.(name + "_Bias") = initializeZeros([numNeurons 1]);
    end
% c.   Generating the parameters for the last fully connect operation.
    sz = [1 numNeurons];
    % The inputs for the last connect operation are also the number of neurons
    numIn = numNeurons; 
    % Initializing the parameters for the connect operation
    parameters.("fc" + numLayers + "_Weights") = initializeHe(sz,numIn,seed);
    seed = seed + 1;
    parameters.("fc" + numLayers + "_Bias") = initializeZeros([1 1]);
end

function [parameters,output,exitflag] = PINN_HE_Dirichlet(in,parameters,options,c)
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
    dlXIC,dlTIC,dlUIC,dlXBC,dlTBC,dlUBC,parameterNames,parameterSizes,c);
% Run the fmincon algorithm and obtain the total runtime of the algorithm
tic
%[x,fval,exitflag,output,lambda,grad,hessian]
[parametersV,fval,exitflag,output,lambda,grad,hessian] = ...
    fmincon(objFun,parametersV,[],[],[],[],[],[],[],options);
toc
% Revert the optimized version of the parameters to a struct. 
parameters = parameterVectorToStruct(parametersV,parameterNames,parameterSizes);
end
